## image/addon.mojo — SIMD image processing on RGBA Uint8Arrays
##
## Four functions demonstrating pixel-level SIMD + parallel computation:
##   1. grayscale(rgba, w, h)          → new Uint8Array (luminance)
##   2. brightness(rgba, w, h, factor) → new Uint8Array (multiply + clamp)
##   3. threshold(rgba, w, h, value)   → new Uint8Array (binary B&W)
##   4. blur(rgba, w, h, radius)       → new Uint8Array (separable box blur)
##
## Build:  pixi run bash image/build.sh
## Run:    node image/image.js

from std.algorithm.functional import vectorize
from std.sys import simd_width_of
from std.math import ceildiv
from std.memory import unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.gpu import global_idx
from max.gpu.host import DeviceContext

from napi.types import NapiEnv, NapiValue
from napi.error import throw_js_error, check_status
from napi.bindings import NapiBindings, Bindings, init_bindings
from napi.raw import raw_set_instance_data, raw_get_instance_data
from napi.framework.js_number import JsNumber
from napi.framework.js_int32 import JsInt32
from napi.framework.js_typedarray import JsTypedArray
from napi.framework.js_arraybuffer import JsArrayBuffer
from napi.framework.args import CbArgs
from napi.framework.register import fn_ptr, ModuleBuilder
from napi.framework.runtime import init_async_runtime, parallelize_safe


comptime NUM_WORKERS = 4

# GPU tuning
comptime GPU_BLOCK = 256


# --- GPU state cache (one DeviceContext per addon instance) ------------------
# Stored via napi_set_instance_data at register_module time. See Phase 2a in
# stats/addon.mojo for the pattern rationale.

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
    var data = OpaquePointer[MutAnyOrigin](unsafe_from_address=Int(0))
    _ = raw_get_instance_data(b, env, Pointer(to=data).unsafe_bitcast[NoneType]().as_unsafe_any_origin())
    if Int(data) == 0:
        raise Error("grayscaleGpu requires a GPU (no accelerator found)")
    return data.unsafe_bitcast[GpuState]()


# --- Grayscale kernel ---------------------------------------------------------
# Integer approximation: gray = (77*R + 150*G + 29*B) >> 8
# Treats each RGBA pixel as a UInt32 lane — extract R/G/B via shifts, pack back.

def _grayscale_rows(
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    start_row: Int, end_row: Int, width: Int,
):
    var src32 = src.unsafe_bitcast[UInt32]()
    var dst32 = dst.unsafe_bitcast[UInt32]()
    var base = start_row * width
    var num_pixels = (end_row - start_row) * width
    def compute[w: Int](offset: Int) {imm src32, imm dst32, imm base}:
        var pixels = src32.unsafe_load[width=w](base + offset)
        var r = pixels & SIMD[DType.uint32, w](0xFF)
        var g = (pixels >> 8) & SIMD[DType.uint32, w](0xFF)
        var b_ch = (pixels >> 16) & SIMD[DType.uint32, w](0xFF)
        var a = pixels & SIMD[DType.uint32, w](0xFF000000)
        var gray = (77 * r + 150 * g + 29 * b_ch) >> 8
        dst32.unsafe_store[width=w](base + offset, gray | (gray << 8) | (gray << 16) | a)
    vectorize[simd_width_of[DType.uint32]()](num_pixels, compute)


def _grayscale_parallel(
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    width: Int, height: Int,
):
    def worker(wid: Int) capturing:
        # NOTE: derived inside the worker, not captured. Capturing a post-computed
        # scalar local in a parallelize closure miscompiles on Linux x86_64
        # (dev2026072306): the capture slot reads garbage on the AsyncRT thread.
        # The compiler flags the bad pattern with "assignment to 'X' was never
        # used" at the capture site. See commit message for the full forensics.
        var rows_per = height // NUM_WORKERS
        var s = wid * rows_per
        var e = s + rows_per if wid < NUM_WORKERS - 1 else height
        _grayscale_rows(src, dst, s, e, width)
    parallelize_safe[worker](NUM_WORKERS)


# --- Grayscale GPU kernel -----------------------------------------------------
# Same fixed-point integer algorithm as the CPU kernel. One thread per pixel.
# No shared memory, no reduction — pure elementwise. Metal-safe (integer only).

def _gpu_kernel_grayscale(
    src: Pointer[UInt32, MutAnyOrigin],
    dst: Pointer[UInt32, MutAnyOrigin],
    num_pixels: Int,
):
    var tid = Int(global_idx.x)
    if tid >= num_pixels:
        return
    var pixel = src[unsafe_offset=tid]
    var r = pixel & 0xFF
    var g = (pixel >> 8) & 0xFF
    var b_ch = (pixel >> 16) & 0xFF
    var a = pixel & 0xFF000000
    var gray = (77 * r + 150 * g + 29 * b_ch) >> 8
    dst[unsafe_offset=tid] = gray | (gray << 8) | (gray << 16) | a


def _grayscale_gpu(
    ctx: DeviceContext,
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    width: Int, height: Int,
) raises:
    var num_pixels = width * height
    var num_bytes = num_pixels * 4

    var dev_src = ctx.enqueue_create_buffer[DType.uint32](num_pixels)
    var dev_dst = ctx.enqueue_create_buffer[DType.uint32](num_pixels)
    var host_src = ctx.enqueue_create_host_buffer[DType.uint32](num_pixels)

    unsafe_memcpy(
        dest=host_src.unsafe_ptr().unsafe_bitcast[Byte](),
        src=src,
        count=num_bytes,
    )
    ctx.enqueue_copy(dev_src, host_src)

    var grid = ceildiv(num_pixels, GPU_BLOCK)
    ctx.enqueue_function[_gpu_kernel_grayscale](
        dev_src.unsafe_ptr(),
        dev_dst.unsafe_ptr(),
        num_pixels,
        grid_dim=grid,
        block_dim=GPU_BLOCK,
    )

    var host_dst = ctx.enqueue_create_host_buffer[DType.uint32](num_pixels)
    ctx.enqueue_copy(host_dst, dev_dst)
    ctx.synchronize()

    unsafe_memcpy(
        dest=dst,
        src=host_dst.unsafe_ptr().unsafe_bitcast[Byte](),
        count=num_bytes,
    )


# --- Brightness kernel --------------------------------------------------------
# Fixed-point multiply: factor_fp = UInt32(factor * 256), result = (ch * fp) >> 8
# SIMD clamp replaces per-byte branch.

def _brightness_rows(
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    start_row: Int, end_row: Int, width: Int,
    factor_fp: UInt32,
):
    var src32 = src.unsafe_bitcast[UInt32]()
    var dst32 = dst.unsafe_bitcast[UInt32]()
    var base = start_row * width
    var num_pixels = (end_row - start_row) * width
    def compute[w: Int](offset: Int) {imm src32, imm dst32, imm base, imm factor_fp}:
        var pixels = src32.unsafe_load[width=w](base + offset)
        var r = pixels & SIMD[DType.uint32, w](0xFF)
        var g = (pixels >> 8) & SIMD[DType.uint32, w](0xFF)
        var b_ch = (pixels >> 16) & SIMD[DType.uint32, w](0xFF)
        var a = pixels & SIMD[DType.uint32, w](0xFF000000)
        var fp = SIMD[DType.uint32, w](factor_fp)
        var nr = ((r * fp) >> 8).clamp(0, 255)
        var ng = ((g * fp) >> 8).clamp(0, 255)
        var nb = ((b_ch * fp) >> 8).clamp(0, 255)
        dst32.unsafe_store[width=w](base + offset, nr | (ng << 8) | (nb << 16) | a)
    vectorize[simd_width_of[DType.uint32]()](num_pixels, compute)


def _brightness_parallel(
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    width: Int, height: Int, factor_fp: UInt32,
):
    def worker(wid: Int) capturing:
        # NOTE: derived inside the worker, not captured. Capturing a post-computed
        # scalar local in a parallelize closure miscompiles on Linux x86_64
        # (dev2026072306): the capture slot reads garbage on the AsyncRT thread.
        # The compiler flags the bad pattern with "assignment to 'X' was never
        # used" at the capture site. See commit message for the full forensics.
        var rows_per = height // NUM_WORKERS
        var s = wid * rows_per
        var e = s + rows_per if wid < NUM_WORKERS - 1 else height
        _brightness_rows(src, dst, s, e, width, factor_fp)
    parallelize_safe[worker](NUM_WORKERS)


# --- Threshold kernel ---------------------------------------------------------
# Grayscale then compare: SIMD select outputs 0x00FFFFFF or 0x000000, preserve alpha.

def _threshold_rows(
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    start_row: Int, end_row: Int, width: Int,
    thresh: Byte,
):
    var src32 = src.unsafe_bitcast[UInt32]()
    var dst32 = dst.unsafe_bitcast[UInt32]()
    var base = start_row * width
    var num_pixels = (end_row - start_row) * width
    def compute[w: Int](offset: Int) {imm src32, imm dst32, imm base, imm thresh}:
        var pixels = src32.unsafe_load[width=w](base + offset)
        var r = pixels & SIMD[DType.uint32, w](0xFF)
        var g = (pixels >> 8) & SIMD[DType.uint32, w](0xFF)
        var b_ch = (pixels >> 16) & SIMD[DType.uint32, w](0xFF)
        var a = pixels & SIMD[DType.uint32, w](0xFF000000)
        var gray = (77 * r + 150 * g + 29 * b_ch) >> 8
        # Branchless: (gray - thresh) wraps to large uint32 if gray < thresh,
        # so bit 31 is set. Shift it down to get 0 (above) or 1 (below).
        var below = (gray - SIMD[DType.uint32, w](UInt32(thresh))) >> 31
        var rgb = (SIMD[DType.uint32, w](1) - below) * SIMD[DType.uint32, w](0x00FFFFFF)
        dst32.unsafe_store[width=w](base + offset, rgb | a)
    vectorize[simd_width_of[DType.uint32]()](num_pixels, compute)


def _threshold_parallel(
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    width: Int, height: Int, thresh: Byte,
):
    def worker(wid: Int) capturing:
        # NOTE: derived inside the worker, not captured. Capturing a post-computed
        # scalar local in a parallelize closure miscompiles on Linux x86_64
        # (dev2026072306): the capture slot reads garbage on the AsyncRT thread.
        # The compiler flags the bad pattern with "assignment to 'X' was never
        # used" at the capture site. See commit message for the full forensics.
        var rows_per = height // NUM_WORKERS
        var s = wid * rows_per
        var e = s + rows_per if wid < NUM_WORKERS - 1 else height
        _threshold_rows(src, dst, s, e, width, thresh)
    parallelize_safe[worker](NUM_WORKERS)


# --- Blur kernel --------------------------------------------------------------
# Separable box blur: horizontal pass + vertical pass
# Uses SIMD[uint32, 4] to process all 4 RGBA channels in parallel per pixel.
# Edge handling: clamp indices to [0, dim-1]

def _blur_horizontal_rows(
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    start_row: Int, end_row: Int, width: Int, radius: Int,
):
    var diameter = 2 * radius + 1
    var diam = SIMD[DType.uint32, 4](UInt32(diameter))
    for row in range(start_row, end_row):
        var row_off = row * width * 4
        # Initialize running sum for first pixel — all 4 channels at once
        var running_sum = SIMD[DType.uint32, 4](0)
        for dx in range(-radius, radius + 1):
            var sx = dx
            if sx < 0:
                sx = 0
            if sx >= width:
                sx = width - 1
            running_sum += src.unsafe_load[width=4](row_off + sx * 4).cast[DType.uint32]()
        dst.unsafe_store[width=4](row_off, (running_sum // diam).cast[DType.uint8]())

        # Slide window across row
        for x in range(1, width):
            var add_x = x + radius
            if add_x >= width:
                add_x = width - 1
            running_sum += src.unsafe_load[width=4](row_off + add_x * 4).cast[DType.uint32]()

            var rem_x = x - radius - 1
            if rem_x < 0:
                rem_x = 0
            running_sum -= src.unsafe_load[width=4](row_off + rem_x * 4).cast[DType.uint32]()

            dst.unsafe_store[width=4](row_off + x * 4, (running_sum // diam).cast[DType.uint8]())


def _blur_vertical_cols(
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    start_col: Int, end_col: Int, width: Int, height: Int, radius: Int,
):
    var diameter = 2 * radius + 1
    var diam = SIMD[DType.uint32, 4](UInt32(diameter))
    for col in range(start_col, end_col):
        var col_off = col * 4
        # Initialize running sum for first pixel — all 4 channels at once
        var running_sum = SIMD[DType.uint32, 4](0)
        for dy in range(-radius, radius + 1):
            var sy = dy
            if sy < 0:
                sy = 0
            if sy >= height:
                sy = height - 1
            running_sum += src.unsafe_load[width=4](sy * width * 4 + col_off).cast[DType.uint32]()
        dst.unsafe_store[width=4](col_off, (running_sum // diam).cast[DType.uint8]())

        # Slide window down column
        for y in range(1, height):
            var add_y = y + radius
            if add_y >= height:
                add_y = height - 1
            running_sum += src.unsafe_load[width=4](add_y * width * 4 + col_off).cast[DType.uint32]()

            var rem_y = y - radius - 1
            if rem_y < 0:
                rem_y = 0
            running_sum -= src.unsafe_load[width=4](rem_y * width * 4 + col_off).cast[DType.uint32]()

            dst.unsafe_store[width=4](y * width * 4 + col_off, (running_sum // diam).cast[DType.uint8]())


def _blur_parallel(
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    width: Int, height: Int, radius: Int,
):
    # Temp buffer for intermediate result between passes
    var temp = unsafe_alloc[Byte](width * height * 4).as_unsafe_any_origin()

    # Horizontal pass: src → temp, parallelize across rows
    var rows_per = height // NUM_WORKERS
    def h_worker(wid: Int) capturing:
        var s = wid * rows_per
        var e = s + rows_per if wid < NUM_WORKERS - 1 else height
        _blur_horizontal_rows(src, temp, s, e, width, radius)
    parallelize_safe[h_worker](NUM_WORKERS)

    # Vertical pass: temp → dst, parallelize across columns
    var cols_per = width // NUM_WORKERS
    def v_worker(wid: Int) capturing:
        var s = wid * cols_per
        var e = s + cols_per if wid < NUM_WORKERS - 1 else width
        _blur_vertical_cols(temp, dst, s, e, width, height, radius)
    parallelize_safe[v_worker](NUM_WORKERS)

    temp.unsafe_free()


# --- N-API callbacks ----------------------------------------------------------

def grayscale_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var bindings = CbArgs.get_bindings(env, info)
        var args = CbArgs.get_three(bindings, env, info)
        var ta = JsTypedArray(args[0])
        var width = Int(JsInt32.from_napi_value(bindings, env, args[1]))
        var height = Int(JsInt32.from_napi_value(bindings, env, args[2]))
        var num_bytes = width * height * 4
        var src_ptr = ta.data_ptr(bindings, env)
        var ab = JsArrayBuffer.create(bindings, env, UInt(num_bytes))
        var dst_ptr = ab.data_ptr(bindings, env)
        _grayscale_parallel(src_ptr, dst_ptr, width, height)
        var result_ta = JsTypedArray.create_uint8(bindings, env, ab.value, 0, UInt(num_bytes))
        return result_ta.value
    except:
        throw_js_error(env, "grayscale failed")
        return NapiValue(unsafe_from_address=Int(0))


def grayscale_gpu_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var bindings = CbArgs.get_bindings(env, info)
        var args = CbArgs.get_three(bindings, env, info)
        var ta = JsTypedArray(args[0])
        var width = Int(JsInt32.from_napi_value(bindings, env, args[1]))
        var height = Int(JsInt32.from_napi_value(bindings, env, args[2]))
        var num_bytes = width * height * 4
        var src_ptr = ta.data_ptr(bindings, env)
        var ab = JsArrayBuffer.create(bindings, env, UInt(num_bytes))
        var dst_ptr = ab.data_ptr(bindings, env)
        var state = _get_gpu_state(bindings, env)
        _grayscale_gpu(state[].ctx, src_ptr, dst_ptr, width, height)
        var result_ta = JsTypedArray.create_uint8(bindings, env, ab.value, 0, UInt(num_bytes))
        return result_ta.value
    except:
        throw_js_error(env, "grayscaleGpu failed (no GPU or kernel error)")
        return NapiValue(unsafe_from_address=Int(0))


def brightness_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var bindings = CbArgs.get_bindings(env, info)
        var args = CbArgs.get_four(bindings, env, info)
        var ta = JsTypedArray(args[0])
        var width = Int(JsInt32.from_napi_value(bindings, env, args[1]))
        var height = Int(JsInt32.from_napi_value(bindings, env, args[2]))
        var factor = JsNumber.from_napi_value(bindings, env, args[3])
        var num_bytes = width * height * 4
        var src_ptr = ta.data_ptr(bindings, env)
        var ab = JsArrayBuffer.create(bindings, env, UInt(num_bytes))
        var dst_ptr = ab.data_ptr(bindings, env)
        var factor_fp = UInt32(factor * 256.0)
        _brightness_parallel(src_ptr, dst_ptr, width, height, factor_fp)
        var result_ta = JsTypedArray.create_uint8(bindings, env, ab.value, 0, UInt(num_bytes))
        return result_ta.value
    except:
        throw_js_error(env, "brightness failed")
        return NapiValue(unsafe_from_address=Int(0))


def threshold_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var bindings = CbArgs.get_bindings(env, info)
        var args = CbArgs.get_four(bindings, env, info)
        var ta = JsTypedArray(args[0])
        var width = Int(JsInt32.from_napi_value(bindings, env, args[1]))
        var height = Int(JsInt32.from_napi_value(bindings, env, args[2]))
        var thresh = Byte(JsInt32.from_napi_value(bindings, env, args[3]))
        var num_bytes = width * height * 4
        var src_ptr = ta.data_ptr(bindings, env)
        var ab = JsArrayBuffer.create(bindings, env, UInt(num_bytes))
        var dst_ptr = ab.data_ptr(bindings, env)
        _threshold_parallel(src_ptr, dst_ptr, width, height, thresh)
        var result_ta = JsTypedArray.create_uint8(bindings, env, ab.value, 0, UInt(num_bytes))
        return result_ta.value
    except:
        throw_js_error(env, "threshold failed")
        return NapiValue(unsafe_from_address=Int(0))


def blur_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var bindings = CbArgs.get_bindings(env, info)
        var args = CbArgs.get_four(bindings, env, info)
        var ta = JsTypedArray(args[0])
        var width = Int(JsInt32.from_napi_value(bindings, env, args[1]))
        var height = Int(JsInt32.from_napi_value(bindings, env, args[2]))
        var radius = Int(JsInt32.from_napi_value(bindings, env, args[3]))
        var num_bytes = width * height * 4
        var src_ptr = ta.data_ptr(bindings, env)
        var ab = JsArrayBuffer.create(bindings, env, UInt(num_bytes))
        var dst_ptr = ab.data_ptr(bindings, env)
        _blur_parallel(src_ptr, dst_ptr, width, height, radius)
        var result_ta = JsTypedArray.create_uint8(bindings, env, ab.value, 0, UInt(num_bytes))
        return result_ta.value
    except:
        throw_js_error(env, "blur failed")
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

    # Cache a DeviceContext if a GPU is available. Skip silently if not.
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
        pass

    var gray_ref = grayscale_fn
    var gray_gpu_ref = grayscale_gpu_fn
    var bright_ref = brightness_fn
    var thresh_ref = threshold_fn
    var blur_ref = blur_fn

    try:
        var m = ModuleBuilder(env, exports, cb_data)
        m.method("grayscale", fn_ptr(gray_ref))
        m.method("grayscaleGpu", fn_ptr(gray_gpu_ref))
        m.method("brightness", fn_ptr(bright_ref))
        m.method("threshold", fn_ptr(thresh_ref))
        m.method("blur", fn_ptr(blur_ref))
        m.flush()
    except:
        pass

    return exports
