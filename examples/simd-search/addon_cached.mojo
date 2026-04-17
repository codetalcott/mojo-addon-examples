## simd-search/addon_cached.mojo — persistent device buffer prototype
##
## Phase 3a.1 prototype: upload a Buffer to the GPU once, query it many times
## without paying H2D copy + device allocation on every call. Tests the
## hypothesis that amortizing PCIe transfer across N calls flips the Phase 2c
## finding that GPU loses to CPU SIMD for single-shot byte scanning.
##
## Exports (as a separate .node from the main search.node):
##   loadGpu(buf)           -> External handle (GC-finalized; GPU memory freed
##                             when the JS handle becomes unreachable)
##   countByteHandle(h, b)  -> count of byte b in the cached buffer
##   releaseGpu(h)          -> marks handle invalid; memory freed on next GC
##
## Build: pixi run bash simd-search/build_cached.sh

from std.math import ceildiv
from std.memory import alloc, memcpy, stack_allocation
from std.gpu import thread_idx, block_idx, barrier
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from napi.types import NapiEnv, NapiValue
from napi.error import throw_js_error
from napi.bindings import NapiBindings, Bindings, init_bindings
from napi.framework.instance_data import set_instance_data, get_instance_data
from napi.framework.js_number import JsNumber
from napi.framework.js_int32 import JsInt32
from napi.framework.js_buffer import JsBuffer
from napi.framework.js_typedarray import JsTypedArray
from napi.framework.js_external import JsExternal
from napi.framework.args import CbArgs
from napi.framework.register import fn_ptr, ModuleBuilder
from napi.framework.runtime import init_async_runtime


# --- GPU tuning -------------------------------------------------------------

comptime GPU_BLOCK = 256
comptime GPU_ELEMS_PER_THREAD = 16
comptime GPU_CHUNK = GPU_BLOCK * GPU_ELEMS_PER_THREAD


# --- GpuState: cached DeviceContext (shared with other addons' pattern) -----

struct GpuState(Movable):
    var ctx: DeviceContext

    def __init__(out self, var ctx: DeviceContext):
        self.ctx = ctx^


def _get_gpu_state(
    b: Bindings, env: NapiEnv
) raises -> UnsafePointer[GpuState, MutAnyOrigin]:
    try:
        return get_instance_data[GpuState](b, env)
    except:
        raise Error("loadGpu requires a GPU (no accelerator found)")


# --- CachedBuffer: device-resident byte buffer + reusable partial sums ------
#
# Holds a DeviceBuffer containing the uploaded bytes, a pre-allocated
# DeviceBuffer for per-block partial sums, and a pinned HostBuffer for the
# D2H destination. All three are sized at loadGpu time and reused on every
# countByteHandle call — no per-call allocation, no per-call H2D copy.
#
# Lifetime: the CachedBuffer is wrapped via JsExternal.create_typed, which
# installs a generic finalizer that destroys the struct on GC (in turn
# destroying the contained DeviceBuffers/HostBuffer via normal destructor
# chain). `released` is a tombstone flag set by releaseGpu to reject
# subsequent queries; the actual memory free still waits for GC.

struct CachedBuffer(Movable):
    var data: DeviceBuffer[DType.uint8]
    var partial: DeviceBuffer[DType.uint32]
    var host_partial: HostBuffer[DType.uint32]
    var size: Int
    var num_blocks: Int
    var released: Bool

    def __init__(
        out self,
        var data: DeviceBuffer[DType.uint8],
        var partial: DeviceBuffer[DType.uint32],
        var host_partial: HostBuffer[DType.uint32],
        size: Int,
        num_blocks: Int,
    ):
        self.data = data^
        self.partial = partial^
        self.host_partial = host_partial^
        self.size = size
        self.num_blocks = num_blocks
        self.released = False


# --- GPU kernel (identical to the non-cached addon) -------------------------
# Each thread reads GPU_ELEMS_PER_THREAD bytes, counts matches locally, writes
# to shared memory, then block-wide tree reduction. One partial per block.

def _gpu_kernel_count_byte(
    data: UnsafePointer[Byte, MutAnyOrigin],
    partial: UnsafePointer[UInt32, MutAnyOrigin],
    target: UInt32,
    size: Int,
):
    var s_count = stack_allocation[
        GPU_BLOCK, Scalar[DType.uint32], address_space=AddressSpace.SHARED
    ]()

    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var base = bid * GPU_CHUNK + tid

    var local: UInt32 = 0
    for i in range(GPU_ELEMS_PER_THREAD):
        var idx = base + i * GPU_BLOCK
        if idx < size:
            if UInt32(data[idx]) == target:
                local += 1

    s_count[tid] = local
    barrier()

    var step = GPU_BLOCK // 2
    while step > 0:
        if tid < step:
            s_count[tid] = s_count[tid] + s_count[tid + step]
        barrier()
        step //= 2

    if tid == 0:
        partial[bid] = s_count[0]


# --- Host pointer extraction (duplicated from addon.mojo) -------------------

def _get_data_ptr(
    b: Bindings, env: NapiEnv, val: NapiValue
) raises -> UnsafePointer[Byte, MutAnyOrigin]:
    if JsBuffer.is_buffer(b, env, val):
        return JsBuffer(val).data_ptr(b, env)
    if JsTypedArray.is_typedarray(b, env, val):
        return JsTypedArray(val).data_ptr(b, env)
    raise Error("expected Buffer or Uint8Array")


def _get_data_len(b: Bindings, env: NapiEnv, val: NapiValue) raises -> Int:
    if JsBuffer.is_buffer(b, env, val):
        return Int(JsBuffer(val).length(b, env))
    if JsTypedArray.is_typedarray(b, env, val):
        return Int(JsTypedArray(val).length(b, env))
    raise Error("expected Buffer or Uint8Array")


# --- loadGpu: one-shot H2D upload into reusable device buffers --------------

def _load_gpu(
    ctx: DeviceContext,
    host_data: UnsafePointer[Byte, MutAnyOrigin],
    size: Int,
) raises -> CachedBuffer:
    var num_blocks = ceildiv(size, GPU_CHUNK)

    var dev_data = ctx.enqueue_create_buffer[DType.uint8](size)

    var staging = ctx.enqueue_create_host_buffer[DType.uint8](size)
    memcpy(dest=staging.unsafe_ptr(), src=host_data, count=size)
    ctx.enqueue_copy(dev_data, staging)

    var dev_partial = ctx.enqueue_create_buffer[DType.uint32](num_blocks)
    var host_partial = ctx.enqueue_create_host_buffer[DType.uint32](num_blocks)

    ctx.synchronize()

    return CachedBuffer(
        dev_data^, dev_partial^, host_partial^, size, num_blocks
    )


def load_gpu_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_one(env, info)
        var b = r.b
        var host_ptr = _get_data_ptr(b, env, r.arg0)
        var size = _get_data_len(b, env, r.arg0)
        var state = _get_gpu_state(b, env)

        var cb_val = _load_gpu(state[].ctx, host_ptr, size)
        return JsExternal.create_typed(b, env, cb_val^).value
    except:
        throw_js_error(env, "loadGpu failed (no GPU or upload error)")
        return NapiValue()


# --- countByteHandle: query cached buffer -----------------------------------

def _count_byte_cached(
    ctx: DeviceContext,
    cb: UnsafePointer[CachedBuffer, MutAnyOrigin],
    target: Byte,
) raises -> Int:
    ctx.enqueue_function[_gpu_kernel_count_byte, _gpu_kernel_count_byte](
        cb[].data.unsafe_ptr(),
        cb[].partial.unsafe_ptr(),
        UInt32(target),
        cb[].size,
        grid_dim=cb[].num_blocks,
        block_dim=GPU_BLOCK,
    )
    ctx.enqueue_copy(cb[].host_partial, cb[].partial)
    ctx.synchronize()

    var ptr = cb[].host_partial.unsafe_ptr()
    var total: Int = 0
    for i in range(cb[].num_blocks):
        total += Int(ptr[i])
    return total


def count_byte_handle_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_two(env, info)
        var b = r.b
        var cb = JsExternal.get_typed[CachedBuffer](
            b, env, r.arg0, "countByteHandle"
        )
        if cb[].released:
            raise Error("countByteHandle: handle has been released")
        var target = Byte(JsInt32.from_napi_value(b, env, r.arg1))
        var state = _get_gpu_state(b, env)
        var count = _count_byte_cached(state[].ctx, cb, target)
        return JsNumber.create(b, env, Float64(count)).value
    except:
        throw_js_error(env, "countByteHandle failed")
        return NapiValue()


# --- releaseGpu: tombstone a handle (actual free is GC-driven) --------------

def release_gpu_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_one(env, info)
        var b = r.b
        var cb = JsExternal.get_typed[CachedBuffer](
            b, env, r.arg0, "releaseGpu"
        )
        cb[].released = True
        return JsNumber.create(b, env, 0.0).value
    except:
        throw_js_error(env, "releaseGpu failed")
        return NapiValue()


# --- Module entry point -----------------------------------------------------

@export("napi_register_module_v1", ABI="C")
def register_module(env: NapiEnv, exports: NapiValue) -> NapiValue:
    try:
        init_async_runtime()
    except:
        pass

    var bindings_ptr = alloc[NapiBindings](1)
    try:
        var bindings = NapiBindings()
        init_bindings(bindings)
        bindings_ptr.init_pointee_move(bindings^)
    except:
        bindings_ptr.free()
        return exports
    var cb_data = bindings_ptr.bitcast[NoneType]()

    try:
        var ctx = DeviceContext()
        set_instance_data(bindings_ptr, env, GpuState(ctx^))
    except:
        pass

    var lg_ref = load_gpu_fn
    var cbh_ref = count_byte_handle_fn
    var rg_ref = release_gpu_fn

    try:
        var m = ModuleBuilder(env, exports, cb_data)
        m.method("loadGpu", fn_ptr(lg_ref))
        m.method("countByteHandle", fn_ptr(cbh_ref))
        m.method("releaseGpu", fn_ptr(rg_ref))
        m.flush()
    except:
        pass

    return exports
