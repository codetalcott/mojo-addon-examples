## image/addon_cached.mojo — persistent device buffer prototype for grayscale
##
## Phase 3b.1 prototype: upload an RGBA image to the GPU once, run the
## grayscale kernel against it many times without paying H2D copy or device
## allocation per call. Unlike countByte (whose output is a scalar), grayscale
## is a transform — every call still pays full D2H for the result image. So
## the cached-vs-one-shot gap is smaller than for reductions: amortizing just
## the one-time H2D + allocation, not the per-call transfer.
##
## API (as a separate .node from image.node):
##   loadImageGpu(rgba, w, h)       -> External handle (GC-finalized)
##   grayscaleHandle(h, dstU8)       -> fills caller-owned Uint8Array in place
##   releaseImageGpu(h)              -> marks handle invalid; memory freed on GC
##
## Build: pixi run bash image/build_cached.sh

from std.math import ceildiv
from std.memory import alloc, memcpy
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from napi.types import NapiEnv, NapiValue
from napi.error import throw_js_error
from napi.bindings import NapiBindings, Bindings, init_bindings
from napi.framework.instance_data import set_instance_data, get_instance_data
from napi.framework.js_number import JsNumber
from napi.framework.js_int32 import JsInt32
from napi.framework.js_typedarray import JsTypedArray
from napi.framework.js_external import JsExternal
from napi.framework.args import CbArgs
from napi.framework.register import fn_ptr, ModuleBuilder
from napi.framework.runtime import init_async_runtime


# --- GPU tuning -------------------------------------------------------------

comptime GPU_BLOCK = 256


# --- GpuState: cached DeviceContext -----------------------------------------

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
        raise Error("loadImageGpu requires a GPU (no accelerator found)")


# --- CachedImage: device-resident RGBA buffers + pinned D2H destination -----
#
# Holds the persistent device input (populated once via loadImageGpu), the
# persistent device output buffer (written fresh by every grayscaleHandle
# call), and a pinned host buffer serving as the D2H landing pad. Only the
# three persistent buffers live in the struct — the H2D staging buffer used
# during loadImageGpu is scope-local and dropped after its upload completes.
#
# Grayscale needs no (w, h) geometry; if future kernels (blur) need stride
# info, add a separate CachedImage2D struct rather than retrofitting.

struct CachedImage(Movable):
    var src: DeviceBuffer[DType.uint32]
    var dst: DeviceBuffer[DType.uint32]
    var host_dst: HostBuffer[DType.uint32]
    var num_pixels: Int
    var num_bytes: Int
    var released: Bool

    def __init__(
        out self,
        var src: DeviceBuffer[DType.uint32],
        var dst: DeviceBuffer[DType.uint32],
        var host_dst: HostBuffer[DType.uint32],
        num_pixels: Int,
    ):
        self.src = src^
        self.dst = dst^
        self.host_dst = host_dst^
        self.num_pixels = num_pixels
        self.num_bytes = num_pixels * 4
        self.released = False


# --- GPU kernel (identical to the non-cached addon) -------------------------
# One thread per pixel, reads packed UInt32 RGBA, writes packed UInt32 result.
# Fixed-point luma: (77*R + 150*G + 29*B) >> 8. Alpha preserved.

def _gpu_kernel_grayscale(
    src: UnsafePointer[UInt32, MutAnyOrigin],
    dst: UnsafePointer[UInt32, MutAnyOrigin],
    num_pixels: Int,
):
    var tid = Int(global_idx.x)
    if tid >= num_pixels:
        return
    var pixel = src[tid]
    var r = pixel & 0xFF
    var g = (pixel >> 8) & 0xFF
    var b_ch = (pixel >> 16) & 0xFF
    var a = pixel & 0xFF000000
    var gray = (77 * r + 150 * g + 29 * b_ch) >> 8
    dst[tid] = gray | (gray << 8) | (gray << 16) | a


# --- loadImageGpu: one-shot H2D upload into reusable device buffers ---------

def _load_image_gpu(
    ctx: DeviceContext,
    src_bytes: UnsafePointer[Byte, MutAnyOrigin],
    num_pixels: Int,
) raises -> CachedImage:
    var num_bytes = num_pixels * 4

    # Persistent device input buffer.
    var dev_src = ctx.enqueue_create_buffer[DType.uint32](num_pixels)

    # Ephemeral pinned staging buffer — dropped on return.
    var staging = ctx.enqueue_create_host_buffer[DType.uint32](num_pixels)
    memcpy(
        dest=staging.unsafe_ptr().bitcast[Byte](),
        src=src_bytes,
        count=num_bytes,
    )
    ctx.enqueue_copy(dev_src, staging)

    # Persistent device output + pinned host D2H landing pad.
    var dev_dst = ctx.enqueue_create_buffer[DType.uint32](num_pixels)
    var host_dst = ctx.enqueue_create_host_buffer[DType.uint32](num_pixels)

    # Block until H2D copy completes so `staging` is safe to drop.
    ctx.synchronize()

    return CachedImage(dev_src^, dev_dst^, host_dst^, num_pixels)


def load_image_gpu_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var b = CbArgs.get_bindings(env, info)
        var args = CbArgs.get_three(b, env, info)
        var ta = JsTypedArray(args[0])
        var width = Int(JsInt32.from_napi_value(b, env, args[1]))
        var height = Int(JsInt32.from_napi_value(b, env, args[2]))
        var num_pixels = width * height
        var src_ptr = ta.data_ptr(b, env)
        var state = _get_gpu_state(b, env)

        var ci_val = _load_image_gpu(state[].ctx, src_ptr, num_pixels)
        return JsExternal.create_typed(b, env, ci_val^).value
    except:
        throw_js_error(env, "loadImageGpu failed (no GPU or upload error)")
        return NapiValue()


# --- grayscaleHandle: query cached image ------------------------------------

def _grayscale_cached(
    ctx: DeviceContext,
    ci: UnsafePointer[CachedImage, MutAnyOrigin],
    dst_bytes: UnsafePointer[Byte, MutAnyOrigin],
) raises:
    var grid = ceildiv(ci[].num_pixels, GPU_BLOCK)
    ctx.enqueue_function[_gpu_kernel_grayscale, _gpu_kernel_grayscale](
        ci[].src.unsafe_ptr(),
        ci[].dst.unsafe_ptr(),
        ci[].num_pixels,
        grid_dim=grid,
        block_dim=GPU_BLOCK,
    )
    ctx.enqueue_copy(ci[].host_dst, ci[].dst)
    ctx.synchronize()

    memcpy(
        dest=dst_bytes,
        src=ci[].host_dst.unsafe_ptr().bitcast[Byte](),
        count=ci[].num_bytes,
    )


def grayscale_handle_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_two(env, info)
        var b = r.b
        var ci = JsExternal.get_typed[CachedImage](
            b, env, r.arg0, "grayscaleHandle"
        )
        if ci[].released:
            raise Error("grayscaleHandle: handle has been released")

        var dst_ta = JsTypedArray(r.arg1)
        var dst_len = Int(dst_ta.length(b, env))
        if dst_len < ci[].num_bytes:
            raise Error("grayscaleHandle: dst buffer too small")
        var dst_ptr = dst_ta.data_ptr(b, env)

        var state = _get_gpu_state(b, env)
        _grayscale_cached(state[].ctx, ci, dst_ptr)
        return JsNumber.create(b, env, 0.0).value
    except:
        throw_js_error(env, "grayscaleHandle failed")
        return NapiValue()


# --- releaseImageGpu: tombstone a handle ------------------------------------

def release_image_gpu_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_one(env, info)
        var b = r.b
        var ci = JsExternal.get_typed[CachedImage](
            b, env, r.arg0, "releaseImageGpu"
        )
        ci[].released = True
        return JsNumber.create(b, env, 0.0).value
    except:
        throw_js_error(env, "releaseImageGpu failed")
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

    # Cache a DeviceContext if a GPU is available.
    try:
        var ctx = DeviceContext()
        set_instance_data(bindings_ptr, env, GpuState(ctx^))
    except:
        pass

    var lig_ref = load_image_gpu_fn
    var gh_ref = grayscale_handle_fn
    var rig_ref = release_image_gpu_fn

    try:
        var m = ModuleBuilder(env, exports, cb_data)
        m.method("loadImageGpu", fn_ptr(lig_ref))
        m.method("grayscaleHandle", fn_ptr(gh_ref))
        m.method("releaseImageGpu", fn_ptr(rig_ref))
        m.flush()
    except:
        pass

    return exports
