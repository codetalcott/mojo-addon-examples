## stats/addon.mojo — SIMD statistics on Float64Arrays
##
## Two functions demonstrating SIMD aggregate computation:
##   1. stats(data)           → {mean, stddev, min, max, p50, p95, p99}
##   2. histogram(data, bins) → Float64Array of bin counts
##
## Build:  pixi run bash stats/build.sh
## Run:    node stats/stats.js

from std.algorithm.functional import vectorize
from std.sys import simd_width_of
from std.math import sqrt, ceildiv
from std.memory import unsafe_memcpy, stack_allocation
from std.memory.alloc import unsafe_alloc
from std.gpu import global_idx, thread_idx, block_idx
from max.gpu import barrier
from std.memory import AddressSpace
from max.gpu.host import DeviceContext

from napi.types import NapiEnv, NapiValue
from napi.error import throw_js_error, check_status
from napi.bindings import NapiBindings, Bindings, init_bindings
from napi.raw import raw_set_instance_data, raw_get_instance_data
from napi.framework.js_number import JsNumber
from napi.framework.js_int32 import JsInt32
from napi.framework.js_object import JsObject
from napi.framework.js_typedarray import JsTypedArray
from napi.framework.js_arraybuffer import JsArrayBuffer
from napi.framework.args import CbArgs
from napi.framework.register import fn_ptr, ModuleBuilder
from napi.framework.runtime import init_async_runtime, parallelize_safe


# --- SIMD sum/min/max in one pass ---------------------------------------------

comptime PARALLEL_THRESHOLD = 4096
comptime NUM_WORKERS = 4

# GPU tuning
comptime GPU_BLOCK = 256
comptime GPU_ELEMS_PER_THREAD = 8
comptime GPU_CHUNK = GPU_BLOCK * GPU_ELEMS_PER_THREAD  # elements per block


# --- GPU state cache (one DeviceContext per addon instance) ------------------
# Stored via napi_set_instance_data at register_module time and retrieved in
# every GPU callback. Avoids the Metal-resource leak that happens when a fresh
# DeviceContext is created per call.

struct GpuState(Movable):
    var ctx: DeviceContext

    def __init__(out self, var ctx: DeviceContext):
        self.ctx = ctx^


def _gpu_state_finalize(
    env: NapiEnv,
    data: OpaquePointer[MutAnyOrigin],
    hint: OpaquePointer[MutAnyOrigin],
):
    var ptr = data.unsafe_bitcast[GpuState]()
    ptr.unsafe_deinit_pointee()
    ptr.unsafe_free()


def _get_gpu_state(
    b: Bindings, env: NapiEnv
) raises -> Pointer[GpuState, MutAnyOrigin]:
    """Fetch the cached GpuState pointer, or raise if GPU unavailable."""
    var data = OpaquePointer[MutAnyOrigin](unsafe_from_address=Int(0))
    _ = raw_get_instance_data(b, env, Pointer(to=data).unsafe_bitcast[NoneType]().as_unsafe_any_origin())
    if Int(data) == 0:
        raise Error("statsGpu requires a GPU (no accelerator found)")
    return data.unsafe_bitcast[GpuState]()

def _simd_sum_min_max(
    data: Pointer[Float64, MutAnyOrigin], start: Int, end: Int
) -> Array[Float64, 3]:
    # Returns [sum, min, max]
    var vsum: Float64 = 0.0
    var vmin: Float64 = data[unsafe_offset=start]
    var vmax: Float64 = data[unsafe_offset=start]
    var base = start
    def compute[width: Int](offset: Int) {mut vsum, mut vmin, mut vmax, imm data, imm base}:
        var chunk = data.unsafe_load[width=width](base + offset)
        vsum += chunk.reduce_add()
        var cmin = chunk.reduce_min()
        var cmax = chunk.reduce_max()
        if cmin < vmin:
            vmin = cmin
        if cmax > vmax:
            vmax = cmax
    vectorize[simd_width_of[DType.float64]()](end - start, compute)
    var result = Array[Float64, 3](fill=vsum)
    result[1] = vmin
    result[2] = vmax
    return result^


def _parallel_sum_min_max(
    data: Pointer[Float64, MutAnyOrigin], size: Int
) -> Array[Float64, 3]:
    if size < PARALLEL_THRESHOLD:
        return _simd_sum_min_max(data, 0, size)
    var p_sum = unsafe_alloc[Float64](NUM_WORKERS)
    var p_min = unsafe_alloc[Float64](NUM_WORKERS)
    var p_max = unsafe_alloc[Float64](NUM_WORKERS)
    def worker(wid: Int) capturing:
        # NOTE: derived inside the worker, not captured. Capturing a post-computed
        # scalar local in a parallelize closure miscompiles on Linux x86_64
        # (dev2026072306): the capture slot reads garbage on the AsyncRT thread.
        # The compiler flags the bad pattern with "assignment to 'X' was never
        # used" at the capture site. See commit message for the full forensics.
        var chunk_size = size // NUM_WORKERS
        var s = wid * chunk_size
        var e = s + chunk_size if wid < NUM_WORKERS - 1 else size
        var partial = _simd_sum_min_max(data, s, e)
        p_sum[unsafe_offset=wid] = partial[unsafe_offset=0]
        p_min[unsafe_offset=wid] = partial[unsafe_offset=1]
        p_max[unsafe_offset=wid] = partial[unsafe_offset=2]
    parallelize_safe[worker](NUM_WORKERS)
    var total_sum: Float64 = 0.0
    var total_min: Float64 = p_min[unsafe_offset=0]
    var total_max: Float64 = p_max[unsafe_offset=0]
    for i in range(NUM_WORKERS):
        total_sum += p_sum[unsafe_offset=i]
        if p_min[unsafe_offset=i] < total_min:
            total_min = p_min[unsafe_offset=i]
        if p_max[unsafe_offset=i] > total_max:
            total_max = p_max[unsafe_offset=i]
    p_sum.unsafe_free()
    p_min.unsafe_free()
    p_max.unsafe_free()
    var result = Array[Float64, 3](fill=total_sum)
    result[1] = total_min
    result[2] = total_max
    return result^


# --- SIMD variance pass -------------------------------------------------------

def _simd_sum_sq_diff(
    data: Pointer[Float64, MutAnyOrigin], start: Int, end: Int, mean: Float64
) -> Float64:
    var sum_sq: Float64 = 0.0
    var base = start
    def compute[width: Int](offset: Int) {mut sum_sq, imm data, imm base, imm mean}:
        var chunk = data.unsafe_load[width=width](base + offset)
        var diff = chunk - mean
        sum_sq += (diff * diff).reduce_add()
    vectorize[simd_width_of[DType.float64]()](end - start, compute)
    return sum_sq


def _parallel_sum_sq_diff(
    data: Pointer[Float64, MutAnyOrigin], size: Int, mean: Float64
) -> Float64:
    if size < PARALLEL_THRESHOLD:
        return _simd_sum_sq_diff(data, 0, size, mean)
    var partials = unsafe_alloc[Float64](NUM_WORKERS)
    def worker(wid: Int) capturing:
        # NOTE: derived inside the worker, not captured. Capturing a post-computed
        # scalar local in a parallelize closure miscompiles on Linux x86_64
        # (dev2026072306): the capture slot reads garbage on the AsyncRT thread.
        # The compiler flags the bad pattern with "assignment to 'X' was never
        # used" at the capture site. See commit message for the full forensics.
        var chunk_size = size // NUM_WORKERS
        var s = wid * chunk_size
        var e = s + chunk_size if wid < NUM_WORKERS - 1 else size
        partials[unsafe_offset=wid] = _simd_sum_sq_diff(data, s, e, mean)
    parallelize_safe[worker](NUM_WORKERS)
    var total: Float64 = 0.0
    for i in range(NUM_WORKERS):
        total += partials[unsafe_offset=i]
    partials.unsafe_free()
    return total


# --- Quickselect for percentiles -----------------------------------------------

def _partition(arr: Pointer[Float64, MutAnyOrigin], lo: Int, hi: Int) -> Int:
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


def _quickselect(arr: Pointer[Float64, MutAnyOrigin], size: Int, k: Int) -> Float64:
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


# --- GPU kernels: tree reduction in shared memory ----------------------------
# Apple Metal (and most mobile GPUs) does not support Float64 in compute
# shaders — kernels must run in Float32. We cast Float64 → Float32 during the
# host-to-device copy, do the reductions in Float32, then cast back on the
# host. This costs ~7 decimal digits of precision. For stats on large arrays
# (N ≥ 10K) of well-scaled data the error is negligible; for adversarial
# inputs the user should stick with the CPU `stats()` path.
#
# One block processes GPU_CHUNK elements. Each thread sums GPU_ELEMS_PER_THREAD
# values (strided by GPU_BLOCK), writes to shared memory, then cooperates in a
# tree reduction. Block 0..num_blocks-1 writes one partial each; host finalizes.

def _gpu_kernel_sum_min_max(
    data: Pointer[Float32, MutAnyOrigin],
    partial_sum: Pointer[Float32, MutAnyOrigin],
    partial_min: Pointer[Float32, MutAnyOrigin],
    partial_max: Pointer[Float32, MutAnyOrigin],
    size_i64: Int64,
):
    # Int/UInt are not DevicePassable as of Mojo 26.6 — kernel params must
    # be fixed-width. Convert back to Int for indexing.
    var size = Int(size_i64)
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

    # Neutral extrema for threads that had no data (Float32 max/lowest).
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
    size_i64: Int64,
):
    # Int/UInt are not DevicePassable as of Mojo 26.6 — kernel params must
    # be fixed-width. Convert back to Int for indexing.
    var size = Int(size_i64)
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


# --- GPU host helpers --------------------------------------------------------

def _gpu_sum_min_max(
    ctx: DeviceContext,
    data: Pointer[Float64, MutAnyOrigin],
    size: Int,
) raises -> Array[Float64, 3]:
    var num_blocks = ceildiv(size, GPU_CHUNK)

    var dev_data = ctx.enqueue_create_buffer[DType.float32](size)
    var host_data = ctx.enqueue_create_host_buffer[DType.float32](size)
    var host_ptr = host_data.unsafe_ptr()
    for i in range(size):
        host_ptr[unsafe_offset=i] = Float32(data[unsafe_offset=i])
    ctx.enqueue_copy(dev_data, host_data)

    var dev_psum = ctx.enqueue_create_buffer[DType.float32](num_blocks)
    var dev_pmin = ctx.enqueue_create_buffer[DType.float32](num_blocks)
    var dev_pmax = ctx.enqueue_create_buffer[DType.float32](num_blocks)

    ctx.enqueue_function[_gpu_kernel_sum_min_max](
        dev_data.unsafe_ptr(),
        dev_psum.unsafe_ptr(),
        dev_pmin.unsafe_ptr(),
        dev_pmax.unsafe_ptr(),
        Int64(size),
        grid_dim=num_blocks,
        block_dim=GPU_BLOCK,
    )

    var host_psum = ctx.enqueue_create_host_buffer[DType.float32](num_blocks)
    var host_pmin = ctx.enqueue_create_host_buffer[DType.float32](num_blocks)
    var host_pmax = ctx.enqueue_create_host_buffer[DType.float32](num_blocks)
    ctx.enqueue_copy(host_psum, dev_psum)
    ctx.enqueue_copy(host_pmin, dev_pmin)
    ctx.enqueue_copy(host_pmax, dev_pmax)
    ctx.synchronize()

    var psum_ptr = host_psum.unsafe_ptr()
    var pmin_ptr = host_pmin.unsafe_ptr()
    var pmax_ptr = host_pmax.unsafe_ptr()

    # Final reduction on CPU in Float64 — recovers most of the precision
    # loss of running the per-block sum in Float32.
    var total_sum: Float64 = 0.0
    var total_min: Float64 = Float64(pmin_ptr[unsafe_offset=0])
    var total_max: Float64 = Float64(pmax_ptr[unsafe_offset=0])
    for i in range(num_blocks):
        total_sum += Float64(psum_ptr[unsafe_offset=i])
        var m = Float64(pmin_ptr[unsafe_offset=i])
        if m < total_min:
            total_min = m
        var M = Float64(pmax_ptr[unsafe_offset=i])
        if M > total_max:
            total_max = M

    var result = Array[Float64, 3](fill=total_sum)
    result[1] = total_min
    result[2] = total_max
    return result^


def _gpu_sum_sq_diff(
    ctx: DeviceContext,
    data: Pointer[Float64, MutAnyOrigin],
    size: Int,
    mean: Float64,
) raises -> Float64:
    var num_blocks = ceildiv(size, GPU_CHUNK)

    var dev_data = ctx.enqueue_create_buffer[DType.float32](size)
    var host_data = ctx.enqueue_create_host_buffer[DType.float32](size)
    var host_ptr = host_data.unsafe_ptr()
    for i in range(size):
        host_ptr[unsafe_offset=i] = Float32(data[unsafe_offset=i])
    ctx.enqueue_copy(dev_data, host_data)

    var dev_partial = ctx.enqueue_create_buffer[DType.float32](num_blocks)

    ctx.enqueue_function[_gpu_kernel_sum_sq_diff](
        dev_data.unsafe_ptr(),
        dev_partial.unsafe_ptr(),
        Float32(mean),
        Int64(size),
        grid_dim=num_blocks,
        block_dim=GPU_BLOCK,
    )

    var host_partial = ctx.enqueue_create_host_buffer[DType.float32](num_blocks)
    ctx.enqueue_copy(host_partial, dev_partial)
    ctx.synchronize()

    var ptr = host_partial.unsafe_ptr()
    var total: Float64 = 0.0
    for i in range(num_blocks):
        total += Float64(ptr[unsafe_offset=i])
    return total


def _compute_stats_gpu(
    ctx: DeviceContext,
    data: Pointer[Float64, MutAnyOrigin],
    size: Int,
) raises -> Array[Float64, 7]:
    # Pass 1 + 2 on GPU (Float32 internal), percentiles on CPU.
    var smm = _gpu_sum_min_max(ctx, data, size)
    var mean = smm[0] / Float64(size)

    var sum_sq = _gpu_sum_sq_diff(ctx, data, size, mean)
    var stddev = sqrt(sum_sq / Float64(size))

    var copy = unsafe_alloc[Float64](size).as_unsafe_any_origin()
    for i in range(size):
        copy[unsafe_offset=i] = data[unsafe_offset=i]

    var p50_idx = Int(Float64(size - 1) * 0.5)
    var p95_idx = Int(Float64(size - 1) * 0.95)
    var p99_idx = Int(Float64(size - 1) * 0.99)

    var p50 = _quickselect(copy, size, p50_idx)
    var p95 = _quickselect(copy, size, p95_idx)
    var p99 = _quickselect(copy, size, p99_idx)
    copy.unsafe_free()

    var result = Array[Float64, 7](fill=mean)
    result[1] = stddev
    result[2] = smm[1]
    result[3] = smm[2]
    result[4] = p50
    result[5] = p95
    result[6] = p99
    return result^


# --- Full stats computation ---------------------------------------------------

def _compute_stats(
    data: Pointer[Float64, MutAnyOrigin], size: Int
) -> Array[Float64, 7]:
    # Returns [mean, stddev, min, max, p50, p95, p99]

    # Pass 1: SIMD sum/min/max
    var smm = _parallel_sum_min_max(data, size)
    var mean = smm[0] / Float64(size)

    # Pass 2: SIMD variance
    var sum_sq = _parallel_sum_sq_diff(data, size, mean)
    var stddev = sqrt(sum_sq / Float64(size))

    # Pass 3: Percentiles via quickselect on a copy
    var copy = unsafe_alloc[Float64](size).as_unsafe_any_origin()
    for i in range(size):
        copy[unsafe_offset=i] = data[unsafe_offset=i]

    var p50_idx = Int(Float64(size - 1) * 0.5)
    var p95_idx = Int(Float64(size - 1) * 0.95)
    var p99_idx = Int(Float64(size - 1) * 0.99)

    var p50 = _quickselect(copy, size, p50_idx)
    var p95 = _quickselect(copy, size, p95_idx)
    var p99 = _quickselect(copy, size, p99_idx)
    copy.unsafe_free()

    var result = Array[Float64, 7](fill=mean)
    result[1] = stddev
    result[2] = smm[1]  # min
    result[3] = smm[2]  # max
    result[4] = p50
    result[5] = p95
    result[6] = p99
    return result^


# --- N-API callbacks ----------------------------------------------------------

def stats_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_one(env, info)
        var b = r.b
        if not JsTypedArray.is_typedarray(b, env, r.arg0):
            throw_js_error(env, "stats requires a Float64Array argument")
            return NapiValue(unsafe_from_address=Int(0))
        var ta = JsTypedArray(r.arg0)
        var size = Int(ta.length(b, env))
        if size == 0:
            throw_js_error(env, "stats requires non-empty array")
            return NapiValue(unsafe_from_address=Int(0))
        var ptr = ta.data_ptr(b, env).unsafe_bitcast[Float64]()
        var s = _compute_stats(ptr, size)

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
        throw_js_error(env, "stats failed")
        return NapiValue(unsafe_from_address=Int(0))


def stats_gpu_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_one(env, info)
        var b = r.b
        if not JsTypedArray.is_typedarray(b, env, r.arg0):
            throw_js_error(env, "statsGpu requires a Float64Array argument")
            return NapiValue(unsafe_from_address=Int(0))
        var ta = JsTypedArray(r.arg0)
        var size = Int(ta.length(b, env))
        if size == 0:
            throw_js_error(env, "statsGpu requires non-empty array")
            return NapiValue(unsafe_from_address=Int(0))
        var ptr = ta.data_ptr(b, env).unsafe_bitcast[Float64]()

        # Retrieve the cached DeviceContext (set in register_module). Raises
        # if no GPU was available at load time.
        var state = _get_gpu_state(b, env)
        var s = _compute_stats_gpu(state[].ctx, ptr, size)

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
        throw_js_error(env, "statsGpu failed")
        return NapiValue(unsafe_from_address=Int(0))


def histogram_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_two(env, info)
        var b = r.b
        if not JsTypedArray.is_typedarray(b, env, r.arg0):
            throw_js_error(env, "histogram requires a Float64Array and bin count")
            return NapiValue(unsafe_from_address=Int(0))
        var ta = JsTypedArray(r.arg0)
        var size = Int(ta.length(b, env))
        var bins = Int(JsInt32.from_napi_value(b, env, r.arg1))
        if size == 0 or bins <= 0:
            throw_js_error(env, "histogram requires non-empty array and positive bins")
            return NapiValue(unsafe_from_address=Int(0))
        var ptr = ta.data_ptr(b, env).unsafe_bitcast[Float64]()

        # Find min/max via SIMD
        var smm = _parallel_sum_min_max(ptr, size)
        var vmin = smm[1]
        var vmax = smm[2]

        # Bin the data
        var counts = unsafe_alloc[Float64](bins)
        for i in range(bins):
            counts[unsafe_offset=i] = 0.0
        var range_val = vmax - vmin
        if range_val == 0.0:
            counts[unsafe_offset=0] = Float64(size)
        else:
            for i in range(size):
                var idx = Int((ptr[unsafe_offset=i] - vmin) / range_val * Float64(bins))
                if idx >= bins:
                    idx = bins - 1
                counts[unsafe_offset=idx] += 1.0

        # Create Float64Array result
        var ab = JsArrayBuffer.create(b, env, UInt(bins * 8))
        var ab_ptr = ab.data_ptr(b, env).unsafe_bitcast[Float64]()
        for i in range(bins):
            ab_ptr[unsafe_offset=i] = counts[unsafe_offset=i]
        counts.unsafe_free()
        var result_ta = JsTypedArray.create_float64(b, env, ab.value, 0, UInt(bins))
        return result_ta.value
    except:
        throw_js_error(env, "histogram failed")
        return NapiValue(unsafe_from_address=Int(0))


# --- Module entry point -------------------------------------------------------

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

    # Try to create and cache a DeviceContext. If no GPU is present this
    # raises and we skip the caching — statsGpu will then throw on invocation.
    try:
        var ctx = DeviceContext()
        var state_ptr = unsafe_alloc[GpuState](1)
        state_ptr.unsafe_write(GpuState(ctx^))
        var fin_ref = _gpu_state_finalize
        var fin_ptr = Pointer(to=fin_ref).unsafe_bitcast[
            OpaquePointer[MutAnyOrigin]
        ]()[]
        _ = raw_set_instance_data(
            bindings_ptr.as_unsafe_any_origin(),
            env,
            state_ptr.unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
            fin_ptr,
            OpaquePointer[MutAnyOrigin](unsafe_from_address=Int(0)),
        )
    except:
        pass  # No GPU — CPU-only mode, statsGpu will throw on call.

    var stats_ref = stats_fn
    var stats_gpu_ref = stats_gpu_fn
    var hist_ref = histogram_fn

    try:
        var m = ModuleBuilder(env, exports, cb_data)
        m.method("stats", fn_ptr(stats_ref))
        m.method("statsGpu", fn_ptr(stats_gpu_ref))
        m.method("histogram", fn_ptr(hist_ref))
        m.flush()
    except:
        pass

    return exports
