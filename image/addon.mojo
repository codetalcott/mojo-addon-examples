## image/addon.mojo — SIMD image processing on RGBA Uint8Arrays
##
## Four functions demonstrating pixel-level SIMD + parallel computation:
##   1. grayscale(rgba, w, h)          → new Uint8Array (luminance)
##   2. brightness(rgba, w, h, factor) → new Uint8Array (multiply + clamp)
##   3. threshold(rgba, w, h, value)   → new Uint8Array (binary B&W)
##   4. blur(rgba, w, h, radius)       → new Uint8Array (separable box blur)
##
## Build:  npm run build:image
## Run:    node image/image.js

from std.memory.alloc import unsafe_alloc

from napi.types import NapiEnv, NapiValue
from napi.error import throw_js_error
from napi.bindings import NapiBindings, Bindings, init_bindings
from napi.framework.js_number import JsNumber
from napi.framework.js_int32 import JsInt32
from napi.framework.js_typedarray import JsTypedArray
from napi.framework.js_arraybuffer import JsArrayBuffer
from napi.framework.args import CbArgs
from napi.framework.register import fn_ptr, ModuleBuilder
from napi.framework.runtime import init_async_runtime, parallelize_safe


comptime NUM_WORKERS = 4

# NOTE on the `capturing` workers below: each one recomputes its own row/column
# split from the function's PARAMETERS rather than capturing a `var` local
# computed outside. A local read only from an implicit `capturing` closure is
# invisible to the compiler's liveness analysis ("assignment was never used"),
# so its store can be eliminated and the closure then reads a garbage stack
# slot — an out-of-bounds row index and a SIGBUS in the host Node process.


# --- Grayscale kernel ---------------------------------------------------------
# Integer approximation: gray = (77*R + 150*G + 29*B) >> 8
# Max value = 77*255 + 150*255 + 29*255 = 65280 — fits UInt16

def _grayscale_rows(
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    start_row: Int, end_row: Int, width: Int,
):
    for row in range(start_row, end_row):
        var row_offset = row * width * 4
        for x in range(width):
            var off = row_offset + x * 4
            var r = UInt16(src[unsafe_offset=off])
            var g = UInt16(src[unsafe_offset= off + 1])
            var b = UInt16(src[unsafe_offset= off + 2])
            var gray = Byte((77 * r + 150 * g + 29 * b) >> 8)
            dst[unsafe_offset=off] = gray
            dst[unsafe_offset= off + 1] = gray
            dst[unsafe_offset= off + 2] = gray
            dst[unsafe_offset= off + 3] = src[unsafe_offset= off + 3]


def _grayscale_parallel(
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    width: Int, height: Int,
):
    def worker(wid: Int) capturing:
        var rows_per = height // NUM_WORKERS
        var s = wid * rows_per
        var e = s + rows_per if wid < NUM_WORKERS - 1 else height
        _grayscale_rows(src, dst, s, e, width)

    parallelize_safe[worker](NUM_WORKERS)


# --- Brightness kernel --------------------------------------------------------
# Fixed-point: factor_fp = UInt16(factor * 256)
# Per byte: min(255, (byte * factor_fp) >> 8)

def _brightness_rows(
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    start_row: Int, end_row: Int, width: Int,
    factor_fp: UInt32,
):
    for row in range(start_row, end_row):
        var row_offset = row * width * 4
        for x in range(width):
            var off = row_offset + x * 4
            for c in range(3):
                var val = (UInt32(src[unsafe_offset= off + c]) * factor_fp) >> 8
                if val > 255:
                    val = 255
                dst[unsafe_offset= off + c] = Byte(val)
            dst[unsafe_offset= off + 3] = src[unsafe_offset= off + 3]


def _brightness_parallel(
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    width: Int, height: Int, factor_fp: UInt32,
):
    def worker(wid: Int) capturing:
        var rows_per = height // NUM_WORKERS
        var s = wid * rows_per
        var e = s + rows_per if wid < NUM_WORKERS - 1 else height
        _brightness_rows(src, dst, s, e, width, factor_fp)

    parallelize_safe[worker](NUM_WORKERS)


# --- Threshold kernel ---------------------------------------------------------
# Grayscale then compare: output 0 or 255 for RGB, preserve alpha

def _threshold_rows(
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    start_row: Int, end_row: Int, width: Int,
    thresh: Byte,
):
    for row in range(start_row, end_row):
        var row_offset = row * width * 4
        for x in range(width):
            var off = row_offset + x * 4
            var r = UInt16(src[unsafe_offset=off])
            var g = UInt16(src[unsafe_offset= off + 1])
            var b = UInt16(src[unsafe_offset= off + 2])
            var gray = Byte((77 * r + 150 * g + 29 * b) >> 8)
            var out_val = Byte(255) if gray >= thresh else Byte(0)
            dst[unsafe_offset=off] = out_val
            dst[unsafe_offset= off + 1] = out_val
            dst[unsafe_offset= off + 2] = out_val
            dst[unsafe_offset= off + 3] = src[unsafe_offset= off + 3]


def _threshold_parallel(
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    width: Int, height: Int, thresh: Byte,
):
    def worker(wid: Int) capturing:
        var rows_per = height // NUM_WORKERS
        var s = wid * rows_per
        var e = s + rows_per if wid < NUM_WORKERS - 1 else height
        _threshold_rows(src, dst, s, e, width, thresh)

    parallelize_safe[worker](NUM_WORKERS)


# --- Blur kernel --------------------------------------------------------------
# Separable box blur: horizontal pass + vertical pass
# Each pass uses a sliding window sum with UInt32 accumulator
# Edge handling: clamp indices to [0, dim-1]

def _blur_horizontal_rows(
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    start_row: Int, end_row: Int, width: Int, radius: Int,
):
    var diameter = 2 * radius + 1
    for row in range(start_row, end_row):
        var row_offset = row * width * 4
        for c in range(4):
            # Initialize running sum for first pixel
            var running_sum: UInt32 = 0
            for dx in range(-radius, radius + 1):
                var sx = dx
                if sx < 0:
                    sx = 0
                if sx >= width:
                    sx = width - 1
                running_sum += UInt32(src[unsafe_offset= row_offset + sx * 4 + c])
            dst[unsafe_offset= row_offset + c] = Byte(running_sum // UInt32(diameter))

            # Slide window across row
            for x in range(1, width):
                # Add right edge
                var add_x = x + radius
                if add_x >= width:
                    add_x = width - 1
                running_sum += UInt32(src[unsafe_offset= row_offset + add_x * 4 + c])

                # Remove left edge
                var rem_x = x - radius - 1
                if rem_x < 0:
                    rem_x = 0
                running_sum -= UInt32(src[unsafe_offset= row_offset + rem_x * 4 + c])

                dst[unsafe_offset= row_offset + x * 4 + c] = Byte(running_sum // UInt32(diameter))


def _blur_vertical_cols(
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    start_col: Int, end_col: Int, width: Int, height: Int, radius: Int,
):
    var diameter = 2 * radius + 1
    for col in range(start_col, end_col):
        for c in range(4):
            # Initialize running sum for first pixel
            var running_sum: UInt32 = 0
            for dy in range(-radius, radius + 1):
                var sy = dy
                if sy < 0:
                    sy = 0
                if sy >= height:
                    sy = height - 1
                running_sum += UInt32(src[unsafe_offset= sy * width * 4 + col * 4 + c])
            dst[unsafe_offset= col * 4 + c] = Byte(running_sum // UInt32(diameter))

            # Slide window down column
            for y in range(1, height):
                var add_y = y + radius
                if add_y >= height:
                    add_y = height - 1
                running_sum += UInt32(src[unsafe_offset= add_y * width * 4 + col * 4 + c])

                var rem_y = y - radius - 1
                if rem_y < 0:
                    rem_y = 0
                running_sum -= UInt32(src[unsafe_offset= rem_y * width * 4 + col * 4 + c])

                dst[unsafe_offset= y * width * 4 + col * 4 + c] = Byte(running_sum // UInt32(diameter))


def _blur_parallel(
    src: Pointer[Byte, MutAnyOrigin],
    dst: Pointer[Byte, MutAnyOrigin],
    width: Int, height: Int, radius: Int,
):
    # Temp buffer for intermediate result between passes. `temp` is also read
    # after both closures (to free it), so it stays live — unlike a split index
    # computed only for the closure, which is why those are computed inside.
    var temp = unsafe_alloc[Byte](width * height * 4).as_unsafe_any_origin()

    # Horizontal pass: src → temp, parallelize across rows
    def h_worker(wid: Int) capturing:
        var rows_per = height // NUM_WORKERS
        var s = wid * rows_per
        var e = s + rows_per if wid < NUM_WORKERS - 1 else height
        _blur_horizontal_rows(src, temp, s, e, width, radius)

    parallelize_safe[h_worker](NUM_WORKERS)

    # Vertical pass: temp → dst, parallelize across columns
    def v_worker(wid: Int) capturing:
        var cols_per = width // NUM_WORKERS
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
        var src_ptr = ta.data_ptr_uint8(bindings, env)
        var ab = JsArrayBuffer.create(bindings, env, UInt(num_bytes))
        var dst_ptr = ab.data_ptr(bindings, env)
        _grayscale_parallel(src_ptr, dst_ptr, width, height)
        var result_ta = JsTypedArray.create_uint8(bindings, env, ab.value, 0, UInt(num_bytes))
        return result_ta.value
    except:
        throw_js_error(env, "grayscale failed")
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
        var src_ptr = ta.data_ptr_uint8(bindings, env)
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
        var src_ptr = ta.data_ptr_uint8(bindings, env)
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
        var src_ptr = ta.data_ptr_uint8(bindings, env)
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
        throw_js_error(env, "image: failed to resolve N-API symbols")
        return exports
    var cb_data = bindings_ptr.unsafe_bitcast[NoneType]().as_unsafe_any_origin()

    var gray_ref = grayscale_fn
    var bright_ref = brightness_fn
    var thresh_ref = threshold_fn
    var blur_ref = blur_fn

    try:
        var m = ModuleBuilder(env, exports, cb_data)
        m.method("grayscale", fn_ptr(gray_ref))
        m.method("brightness", fn_ptr(bright_ref))
        m.method("threshold", fn_ptr(thresh_ref))
        m.method("blur", fn_ptr(blur_ref))
        m.flush()
    except:
        throw_js_error(env, "image: failed to register exports")

    return exports
