## stats/addon_cached.mojo — persistent device buffer prototype for stats
##
## Phase 3b.2 prototype: upload a Float64Array to the GPU once, run the
## two stats kernels against it many times without paying Float64→Float32
## cast, H2D transfer, or device allocation per call.
##
## Key design decisions vs stats/addon.mojo:
##
##  1. Float64→Float32 cast happens ONCE at loadStatsGpu time, on the host,
##     using a vectorized SIMD loop (replaces the scalar loop at addon.mojo
##     lines 327 and 388). Portable across Metal (which can't do Float64 in
##     compute shaders) and CUDA. The primary Phase 3 win is that the cast
##     no longer runs per call — not that it's on the GPU.
##
##  2. Same two-kernel structure as stats/addon.mojo (sum/min/max pass, then
##     sum_sq_diff pass after host-side mean computation). Welford fusion is
##     deferred; the big win is persistence, not the kernel count.
##
##  3. Handle owns a heap Float64 copy of the input so statsHandle(h) takes
##     only the handle. Percentiles run CPU-side quickselect against a copy
##     of that buffer on every call (same as stats/addon.mojo). Memory cost:
##     2× the input size (once in JS, once in the handle).
##
## API (as a separate .node from stats.node):
##   loadStatsGpu(data)    -> External handle (GC-finalized)
##   statsHandle(h)        -> {mean, stddev, min, max, p50, p95, p99}
##   releaseStatsGpu(h)    -> tombstone; memory freed on GC
##
## Build: pixi run bash stats/build_cached.sh

from std.algorithm.functional import vectorize
from std.sys import simd_width_of
from std.math import sqrt, ceildiv
from std.memory import unsafe_memcpy, stack_allocation
from std.memory.alloc import unsafe_alloc
from std.gpu import thread_idx, block_idx
from max.gpu import barrier
from std.memory import AddressSpace
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from napi.types import NapiEnv, NapiValue
from napi.error import throw_js_error
from napi.bindings import NapiBindings, Bindings, init_bindings
from napi.framework.instance_data import set_instance_data, get_instance_data
from napi.framework.js_number import JsNumber
from napi.framework.js_object import JsObject
from napi.framework.js_typedarray import JsTypedArray
from napi.framework.js_external import JsExternal
from napi.framework.args import CbArgs
from napi.framework.register import fn_ptr, ModuleBuilder
from napi.framework.runtime import init_async_runtime


# --- GPU tuning (identical to stats/addon.mojo) ------------------------------

comptime GPU_BLOCK = 256
comptime GPU_ELEMS_PER_THREAD = 8
comptime GPU_CHUNK = GPU_BLOCK * GPU_ELEMS_PER_THREAD


# --- GpuState ----------------------------------------------------------------

struct GpuState(Movable):
    var ctx: DeviceContext

    def __init__(out self, var ctx: DeviceContext):
        self.ctx = ctx^


def _get_gpu_state(
    b: Bindings, env: NapiEnv
) raises -> Pointer[GpuState, MutAnyOrigin]:
    try:
        return get_instance_data[GpuState](b, env)
    except:
        raise Error("loadStatsGpu requires a GPU (no accelerator found)")


# --- Quickselect + partition (copied from stats/addon.mojo) ------------------

def _partition(
    arr: Pointer[Float64, MutAnyOrigin], lo: Int, hi: Int
) -> Int:
    var pivot = arr[unsafe_offset=hi]
    var i = lo
    for j in range(lo, hi):
        if arr[unsafe_offset=j] <= pivot:
            var tmp = arr[unsafe_offset=i]
            arr[unsafe_offset=i] = arr[unsafe_offset=j]
            arr[unsafe_offset=j] = tmp
            i += 1
    var tmp = arr[unsafe_offset=i]
    arr[unsafe_offset=i] = arr[unsafe_offset=hi]
    arr[unsafe_offset=hi] = tmp
    return i


def _quickselect(
    arr: Pointer[Float64, MutAnyOrigin], size: Int, k: Int
) -> Float64:
    var left = 0
    var right = size - 1
    while left < right:
        var pivot_idx = _partition(arr, left, right)
        if pivot_idx == k:
            return arr[unsafe_offset=k]
        elif pivot_idx < k:
            left = pivot_idx + 1
        else:
            right = pivot_idx - 1
    return arr[unsafe_offset=left]


# --- CachedStats: persistent device buffers + heap Float64 copy --------------
#
# The device buffers are populated once in loadStatsGpu (with a vectorized
# Float64→Float32 cast during upload), and reused on every statsHandle call.
# The Float64 copy `data_f64` lets quickselect run CPU-side without requiring
# the caller to hand the original array back in.
#
# Memory (10M elements): ~80 MB (Float64 copy) + ~40 MB (device Float32) + a
# few KB of partials. Cached API users accept this overhead in exchange for
# per-call amortization.

struct CachedStats(Movable):
    var dev_data: DeviceBuffer[DType.float32]
    var dev_psum: DeviceBuffer[DType.float32]
    var dev_pmin: DeviceBuffer[DType.float32]
    var dev_pmax: DeviceBuffer[DType.float32]
    var dev_psq: DeviceBuffer[DType.float32]
    var host_psum: HostBuffer[DType.float32]
    var host_pmin: HostBuffer[DType.float32]
    var host_pmax: HostBuffer[DType.float32]
    var host_psq: HostBuffer[DType.float32]
    @__allow_legacy_any_origin_fields
    var data_f64: Pointer[Float64, MutAnyOrigin]
    var size: Int
    var num_blocks: Int
    var released: Bool

    def __init__(
        out self,
        var dev_data: DeviceBuffer[DType.float32],
        var dev_psum: DeviceBuffer[DType.float32],
        var dev_pmin: DeviceBuffer[DType.float32],
        var dev_pmax: DeviceBuffer[DType.float32],
        var dev_psq: DeviceBuffer[DType.float32],
        var host_psum: HostBuffer[DType.float32],
        var host_pmin: HostBuffer[DType.float32],
        var host_pmax: HostBuffer[DType.float32],
        var host_psq: HostBuffer[DType.float32],
        data_f64: Pointer[Float64, MutAnyOrigin],
        size: Int,
        num_blocks: Int,
    ):
        self.dev_data = dev_data^
        self.dev_psum = dev_psum^
        self.dev_pmin = dev_pmin^
        self.dev_pmax = dev_pmax^
        self.dev_psq = dev_psq^
        self.host_psum = host_psum^
        self.host_pmin = host_pmin^
        self.host_pmax = host_pmax^
        self.host_psq = host_psq^
        self.data_f64 = data_f64
        self.size = size
        self.num_blocks = num_blocks
        self.released = False

    def __deinit__(deinit self):
        # DeviceBuffer/HostBuffer destructors run automatically via field
        # teardown; the only field we own explicitly is the heap Float64 copy.
        self.data_f64.unsafe_free()


# --- GPU kernels (verbatim clone of stats/addon.mojo) ------------------------

def _gpu_kernel_sum_min_max(
    data: Pointer[Float32, MutAnyOrigin],
    partial_sum: Pointer[Float32, MutAnyOrigin],
    partial_min: Pointer[Float32, MutAnyOrigin],
    partial_max: Pointer[Float32, MutAnyOrigin],
    size: Int,
):
    var s_sum = stack_allocation[
        GPU_BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var s_min = stack_allocation[
        GPU_BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var s_max = stack_allocation[
        GPU_BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()

    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var base = bid * GPU_CHUNK + tid

    var local_sum: Float32 = 0.0
    var seeded = False
    var local_min: Float32 = 0.0
    var local_max: Float32 = 0.0
    for i in range(GPU_ELEMS_PER_THREAD):
        var idx = base + i * GPU_BLOCK
        if idx < size:
            var v = data[unsafe_offset=idx]
            local_sum += v
            if not seeded:
                local_min = v
                local_max = v
                seeded = True
            else:
                if v < local_min:
                    local_min = v
                if v > local_max:
                    local_max = v

    if not seeded:
        local_min = 3.4e38
        local_max = -3.4e38

    s_sum[unsafe_offset=tid] = local_sum
    s_min[unsafe_offset=tid] = local_min
    s_max[unsafe_offset=tid] = local_max
    barrier()

    var step = GPU_BLOCK // 2
    while step > 0:
        if tid < step:
            s_sum[unsafe_offset=tid] = s_sum[unsafe_offset=tid] + s_sum[unsafe_offset=tid + step]
            var a = s_min[unsafe_offset=tid]
            var b = s_min[unsafe_offset=tid + step]
            s_min[unsafe_offset=tid] = a if a < b else b
            var c = s_max[unsafe_offset=tid]
            var d = s_max[unsafe_offset=tid + step]
            s_max[unsafe_offset=tid] = c if c > d else d
        barrier()
        step //= 2

    if tid == 0:
        partial_sum[unsafe_offset=bid] = s_sum[unsafe_offset=0]
        partial_min[unsafe_offset=bid] = s_min[unsafe_offset=0]
        partial_max[unsafe_offset=bid] = s_max[unsafe_offset=0]


def _gpu_kernel_sum_sq_diff(
    data: Pointer[Float32, MutAnyOrigin],
    partial: Pointer[Float32, MutAnyOrigin],
    mean: Float32,
    size: Int,
):
    var s_sum = stack_allocation[
        GPU_BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()

    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var base = bid * GPU_CHUNK + tid

    var local_sum: Float32 = 0.0
    for i in range(GPU_ELEMS_PER_THREAD):
        var idx = base + i * GPU_BLOCK
        if idx < size:
            var d = data[unsafe_offset=idx] - mean
            local_sum += d * d

    s_sum[unsafe_offset=tid] = local_sum
    barrier()

    var step = GPU_BLOCK // 2
    while step > 0:
        if tid < step:
            s_sum[unsafe_offset=tid] = s_sum[unsafe_offset=tid] + s_sum[unsafe_offset=tid + step]
        barrier()
        step //= 2

    if tid == 0:
        partial[unsafe_offset=bid] = s_sum[unsafe_offset=0]


# --- Vectorized Float64 → Float32 cast ---------------------------------------
# Replaces the scalar cast loop at stats/addon.mojo:327 and :388. Runs once
# at loadStatsGpu time, not per call. On 10M elements this should be around
# 10 ms (memory-bandwidth bound) vs the scalar original's ~50-100 ms.

def _cast_f64_to_f32_simd(
    src: Pointer[Float64, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    size: Int,
):
    def compute[width: Int](offset: Int) {imm src, imm dst}:
        var chunk = src.unsafe_load[width=width](offset)
        dst.unsafe_store[width=width](offset, chunk.cast[DType.float32]())
    vectorize[simd_width_of[DType.float64]()](size, compute)


# --- loadStatsGpu: one-shot cast + H2D upload --------------------------------

def _load_stats_gpu(
    ctx: DeviceContext,
    host_f64: Pointer[Float64, MutAnyOrigin],
    size: Int,
) raises -> CachedStats:
    var num_blocks = ceildiv(size, GPU_CHUNK)

    # Persistent Float32 device buffer.
    var dev_data = ctx.enqueue_create_buffer[DType.float32](size)

    # Ephemeral pinned staging — vectorized cast writes into it, then H2D
    # uploads to dev_data. Dropped on return.
    var staging = ctx.enqueue_create_host_buffer[DType.float32](size)
    _cast_f64_to_f32_simd(host_f64, staging.unsafe_ptr().as_unsafe_any_origin(), size)
    ctx.enqueue_copy(dev_data, staging)

    # Persistent reusable partial buffers (device + pinned host).
    var dev_psum = ctx.enqueue_create_buffer[DType.float32](num_blocks)
    var dev_pmin = ctx.enqueue_create_buffer[DType.float32](num_blocks)
    var dev_pmax = ctx.enqueue_create_buffer[DType.float32](num_blocks)
    var dev_psq = ctx.enqueue_create_buffer[DType.float32](num_blocks)
    var host_psum = ctx.enqueue_create_host_buffer[DType.float32](num_blocks)
    var host_pmin = ctx.enqueue_create_host_buffer[DType.float32](num_blocks)
    var host_pmax = ctx.enqueue_create_host_buffer[DType.float32](num_blocks)
    var host_psq = ctx.enqueue_create_host_buffer[DType.float32](num_blocks)

    # Block until H2D completes so `staging` is safe to drop.
    ctx.synchronize()

    # Heap Float64 copy for CPU-side percentile quickselect.
    var data_f64 = unsafe_alloc[Float64](size).as_unsafe_any_origin()
    unsafe_memcpy(dest=data_f64, src=host_f64, count=size)

    return CachedStats(
        dev_data^,
        dev_psum^,
        dev_pmin^,
        dev_pmax^,
        dev_psq^,
        host_psum^,
        host_pmin^,
        host_pmax^,
        host_psq^,
        data_f64,
        size,
        num_blocks,
    )


def load_stats_gpu_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_one(env, info)
        var b = r.b
        if not JsTypedArray.is_typedarray(b, env, r.arg0):
            raise Error("loadStatsGpu requires a Float64Array argument")
        var ta = JsTypedArray(r.arg0)
        var size = Int(ta.length(b, env))
        if size == 0:
            raise Error("loadStatsGpu requires non-empty array")
        var ptr = ta.data_ptr(b, env).unsafe_bitcast[Float64]()
        var state = _get_gpu_state(b, env)

        var cs_val = _load_stats_gpu(state[].ctx, ptr, size)
        return JsExternal.create_typed(b, env, cs_val^).value
    except:
        throw_js_error(env, "loadStatsGpu failed (no GPU or upload error)")
        return NapiValue(unsafe_from_address=Int(0))


# --- statsHandle: per-call GPU reductions + host percentiles -----------------

def _stats_cached(
    ctx: DeviceContext,
    cs: Pointer[CachedStats, MutAnyOrigin],
) raises -> Array[Float64, 7]:
    # Pass 1: sum + min + max. Persistent buffers, no alloc, no H2D.
    ctx.enqueue_function[_gpu_kernel_sum_min_max](
        cs[].dev_data.unsafe_ptr(),
        cs[].dev_psum.unsafe_ptr(),
        cs[].dev_pmin.unsafe_ptr(),
        cs[].dev_pmax.unsafe_ptr(),
        cs[].size,
        grid_dim=cs[].num_blocks,
        block_dim=GPU_BLOCK,
    )
    ctx.enqueue_copy(cs[].host_psum, cs[].dev_psum)
    ctx.enqueue_copy(cs[].host_pmin, cs[].dev_pmin)
    ctx.enqueue_copy(cs[].host_pmax, cs[].dev_pmax)
    ctx.synchronize()

    var psum_ptr = cs[].host_psum.unsafe_ptr()
    var pmin_ptr = cs[].host_pmin.unsafe_ptr()
    var pmax_ptr = cs[].host_pmax.unsafe_ptr()

    var total_sum: Float64 = 0.0
    var total_min: Float64 = Float64(pmin_ptr[unsafe_offset=0])
    var total_max: Float64 = Float64(pmax_ptr[unsafe_offset=0])
    for i in range(cs[].num_blocks):
        total_sum += Float64(psum_ptr[unsafe_offset=i])
        var m = Float64(pmin_ptr[unsafe_offset=i])
        if m < total_min:
            total_min = m
        var M = Float64(pmax_ptr[unsafe_offset=i])
        if M > total_max:
            total_max = M

    var mean = total_sum / Float64(cs[].size)

    # Pass 2: sum_sq_diff (requires mean).
    ctx.enqueue_function[_gpu_kernel_sum_sq_diff](
        cs[].dev_data.unsafe_ptr(),
        cs[].dev_psq.unsafe_ptr(),
        Float32(mean),
        cs[].size,
        grid_dim=cs[].num_blocks,
        block_dim=GPU_BLOCK,
    )
    ctx.enqueue_copy(cs[].host_psq, cs[].dev_psq)
    ctx.synchronize()

    var psq_ptr = cs[].host_psq.unsafe_ptr()
    var sum_sq: Float64 = 0.0
    for i in range(cs[].num_blocks):
        sum_sq += Float64(psq_ptr[unsafe_offset=i])
    var stddev = sqrt(sum_sq / Float64(cs[].size))

    # Pass 3: percentiles via quickselect on a scratch copy of the Float64
    # cache. Same as stats/addon.mojo _compute_stats — we don't mutate the
    # persistent data_f64 copy because quickselect is destructive.
    var scratch = unsafe_alloc[Float64](cs[].size).as_unsafe_any_origin()
    unsafe_memcpy(dest=scratch, src=cs[].data_f64, count=cs[].size)

    var p50_idx = Int(Float64(cs[].size - 1) * 0.5)
    var p95_idx = Int(Float64(cs[].size - 1) * 0.95)
    var p99_idx = Int(Float64(cs[].size - 1) * 0.99)

    var p50 = _quickselect(scratch, cs[].size, p50_idx)
    var p95 = _quickselect(scratch, cs[].size, p95_idx)
    var p99 = _quickselect(scratch, cs[].size, p99_idx)
    scratch.unsafe_free()

    var result = Array[Float64, 7](fill=mean)
    result[1] = stddev
    result[2] = total_min
    result[3] = total_max
    result[4] = p50
    result[5] = p95
    result[6] = p99
    return result^


def stats_handle_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_one(env, info)
        var b = r.b
        var cs = JsExternal.get_typed[CachedStats](
            b, env, r.arg0, "statsHandle"
        )
        if cs[].released:
            raise Error("statsHandle: handle has been released")

        var state = _get_gpu_state(b, env)
        var s = _stats_cached(state[].ctx, cs)

        var obj = JsObject.create(b, env)
        obj.set_property(b, env, "mean",   JsNumber.create(b, env, s[0]).value)
        obj.set_property(b, env, "stddev", JsNumber.create(b, env, s[1]).value)
        obj.set_property(b, env, "min",    JsNumber.create(b, env, s[2]).value)
        obj.set_property(b, env, "max",    JsNumber.create(b, env, s[3]).value)
        obj.set_property(b, env, "p50",    JsNumber.create(b, env, s[4]).value)
        obj.set_property(b, env, "p95",    JsNumber.create(b, env, s[5]).value)
        obj.set_property(b, env, "p99",    JsNumber.create(b, env, s[6]).value)
        return obj.value
    except:
        throw_js_error(env, "statsHandle failed")
        return NapiValue(unsafe_from_address=Int(0))


# --- releaseStatsGpu: tombstone handle ---------------------------------------

def release_stats_gpu_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_one(env, info)
        var b = r.b
        var cs = JsExternal.get_typed[CachedStats](
            b, env, r.arg0, "releaseStatsGpu"
        )
        cs[].released = True
        return JsNumber.create(b, env, 0.0).value
    except:
        throw_js_error(env, "releaseStatsGpu failed")
        return NapiValue(unsafe_from_address=Int(0))


# --- Module entry point ------------------------------------------------------

@export("napi_register_module_v1")
def register_module(env: NapiEnv, exports: NapiValue) abi("C") -> NapiValue:
    try:
        init_async_runtime()
    except:
        pass

    var bindings_ptr = unsafe_alloc[NapiBindings](1)
    try:
        var bindings = NapiBindings()
        init_bindings(bindings)
        bindings_ptr.unsafe_write(bindings^)
    except:
        bindings_ptr.unsafe_free()
        return exports
    var cb_data = bindings_ptr.unsafe_bitcast[NoneType]().as_unsafe_any_origin()

    try:
        var ctx = DeviceContext()
        set_instance_data(bindings_ptr.as_unsafe_any_origin(), env, GpuState(ctx^))
    except:
        pass

    var lsg_ref = load_stats_gpu_fn
    var sh_ref = stats_handle_fn
    var rsg_ref = release_stats_gpu_fn

    try:
        var m = ModuleBuilder(env, exports, cb_data)
        m.method("loadStatsGpu", fn_ptr(lsg_ref))
        m.method("statsHandle", fn_ptr(sh_ref))
        m.method("releaseStatsGpu", fn_ptr(rsg_ref))
        m.flush()
    except:
        pass

    return exports
