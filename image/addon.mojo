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

from std.algorithm.functional import parallelize
from std.memory import alloc

from napi.types import NapiEnv, NapiValue
from napi.error import throw_js_error
from napi.bindings import NapiBindings, Bindings, init_bindings
from napi.framework.js_number import JsNumber
from napi.framework.js_int32 import JsInt32
from napi.framework.js_typedarray import JsTypedArray
from napi.framework.js_arraybuffer import JsArrayBuffer
from napi.framework.args import CbArgs
from napi.framework.register import fn_ptr, ModuleBuilder
from napi.framework.runtime import init_async_runtime


comptime NUM_WORKERS = 4


# --- Grayscale kernel ---------------------------------------------------------
# Integer approximation: gray = (77*R + 150*G + 29*B) >> 8
# Max value = 77*255 + 150*255 + 29*255 = 65280 — fits UInt16

fn _grayscale_rows(
    src: UnsafePointer[Byte, MutAnyOrigin],
    dst: UnsafePointer[Byte, MutAnyOrigin],
    start_row: Int, end_row: Int, width: Int,
):
    for row in range(start_row, end_row):
        var row_offset = row * width * 4
        for x in range(width):
            var off = row_offset + x * 4
            var r = UInt16(src[off])
            var g = UInt16(src[off + 1])
            var b = UInt16(src[off + 2])
            var gray = Byte((77 * r + 150 * g + 29 * b) >> 8)
            dst[off] = gray
            dst[off + 1] = gray
            dst[off + 2] = gray
            dst[off + 3] = src[off + 3]


fn _grayscale_parallel(
    src: UnsafePointer[Byte, MutAnyOrigin],
    dst: UnsafePointer[Byte, MutAnyOrigin],
    width: Int, height: Int,
):
    var rows_per = height // NUM_WORKERS
    fn worker(wid: Int) capturing:
        var s = wid * rows_per
        var e = s + rows_per if wid < NUM_WORKERS - 1 else height
        _grayscale_rows(src, dst, s, e, width)
    parallelize[worker](NUM_WORKERS)


# --- Brightness kernel --------------------------------------------------------
# Fixed-point: factor_fp = UInt16(factor * 256)
# Per byte: min(255, (byte * factor_fp) >> 8)

fn _brightness_rows(
    src: UnsafePointer[Byte, MutAnyOrigin],
    dst: UnsafePointer[Byte, MutAnyOrigin],
    start_row: Int, end_row: Int, width: Int,
    factor_fp: UInt32,
):
    for row in range(start_row, end_row):
        var row_offset = row * width * 4
        for x in range(width):
            var off = row_offset + x * 4
            for c in range(3):
                var val = (UInt32(src[off + c]) * factor_fp) >> 8
                if val > 255:
                    val = 255
                dst[off + c] = Byte(val)
            dst[off + 3] = src[off + 3]


fn _brightness_parallel(
    src: UnsafePointer[Byte, MutAnyOrigin],
    dst: UnsafePointer[Byte, MutAnyOrigin],
    width: Int, height: Int, factor_fp: UInt32,
):
    var rows_per = height // NUM_WORKERS
    fn worker(wid: Int) capturing:
        var s = wid * rows_per
        var e = s + rows_per if wid < NUM_WORKERS - 1 else height
        _brightness_rows(src, dst, s, e, width, factor_fp)
    parallelize[worker](NUM_WORKERS)


# --- Threshold kernel ---------------------------------------------------------
# Grayscale then compare: output 0 or 255 for RGB, preserve alpha

fn _threshold_rows(
    src: UnsafePointer[Byte, MutAnyOrigin],
    dst: UnsafePointer[Byte, MutAnyOrigin],
    start_row: Int, end_row: Int, width: Int,
    thresh: Byte,
):
    for row in range(start_row, end_row):
        var row_offset = row * width * 4
        for x in range(width):
            var off = row_offset + x * 4
            var r = UInt16(src[off])
            var g = UInt16(src[off + 1])
            var b = UInt16(src[off + 2])
            var gray = Byte((77 * r + 150 * g + 29 * b) >> 8)
            var out_val = Byte(255) if gray >= thresh else Byte(0)
            dst[off] = out_val
            dst[off + 1] = out_val
            dst[off + 2] = out_val
            dst[off + 3] = src[off + 3]


fn _threshold_parallel(
    src: UnsafePointer[Byte, MutAnyOrigin],
    dst: UnsafePointer[Byte, MutAnyOrigin],
    width: Int, height: Int, thresh: Byte,
):
    var rows_per = height // NUM_WORKERS
    fn worker(wid: Int) capturing:
        var s = wid * rows_per
        var e = s + rows_per if wid < NUM_WORKERS - 1 else height
        _threshold_rows(src, dst, s, e, width, thresh)
    parallelize[worker](NUM_WORKERS)


# --- Blur kernel --------------------------------------------------------------
# Separable box blur: horizontal pass + vertical pass
# Each pass uses a sliding window sum with UInt32 accumulator
# Edge handling: clamp indices to [0, dim-1]

fn _blur_horizontal_rows(
    src: UnsafePointer[Byte, MutAnyOrigin],
    dst: UnsafePointer[Byte, MutAnyOrigin],
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
                running_sum += UInt32(src[row_offset + sx * 4 + c])
            dst[row_offset + c] = Byte(running_sum // UInt32(diameter))

            # Slide window across row
            for x in range(1, width):
                # Add right edge
                var add_x = x + radius
                if add_x >= width:
                    add_x = width - 1
                running_sum += UInt32(src[row_offset + add_x * 4 + c])

                # Remove left edge
                var rem_x = x - radius - 1
                if rem_x < 0:
                    rem_x = 0
                running_sum -= UInt32(src[row_offset + rem_x * 4 + c])

                dst[row_offset + x * 4 + c] = Byte(running_sum // UInt32(diameter))


fn _blur_vertical_cols(
    src: UnsafePointer[Byte, MutAnyOrigin],
    dst: UnsafePointer[Byte, MutAnyOrigin],
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
                running_sum += UInt32(src[sy * width * 4 + col * 4 + c])
            dst[col * 4 + c] = Byte(running_sum // UInt32(diameter))

            # Slide window down column
            for y in range(1, height):
                var add_y = y + radius
                if add_y >= height:
                    add_y = height - 1
                running_sum += UInt32(src[add_y * width * 4 + col * 4 + c])

                var rem_y = y - radius - 1
                if rem_y < 0:
                    rem_y = 0
                running_sum -= UInt32(src[rem_y * width * 4 + col * 4 + c])

                dst[y * width * 4 + col * 4 + c] = Byte(running_sum // UInt32(diameter))


fn _blur_parallel(
    src: UnsafePointer[Byte, MutAnyOrigin],
    dst: UnsafePointer[Byte, MutAnyOrigin],
    width: Int, height: Int, radius: Int,
):
    # Temp buffer for intermediate result between passes
    var temp = alloc[Byte](width * height * 4)

    # Horizontal pass: src → temp, parallelize across rows
    var rows_per = height // NUM_WORKERS
    fn h_worker(wid: Int) capturing:
        var s = wid * rows_per
        var e = s + rows_per if wid < NUM_WORKERS - 1 else height
        _blur_horizontal_rows(src, temp, s, e, width, radius)
    parallelize[h_worker](NUM_WORKERS)

    # Vertical pass: temp → dst, parallelize across columns
    var cols_per = width // NUM_WORKERS
    fn v_worker(wid: Int) capturing:
        var s = wid * cols_per
        var e = s + cols_per if wid < NUM_WORKERS - 1 else width
        _blur_vertical_cols(temp, dst, s, e, width, height, radius)
    parallelize[v_worker](NUM_WORKERS)

    temp.free()


# --- N-API callbacks ----------------------------------------------------------

fn grayscale_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
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
        return NapiValue()


fn brightness_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
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
        return NapiValue()


fn threshold_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
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
        return NapiValue()


fn blur_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
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
        return NapiValue()


# --- Module entry point -------------------------------------------------------

@export("napi_register_module_v1", ABI="C")
fn register_module(env: NapiEnv, exports: NapiValue) -> NapiValue:
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
        pass

    return exports
