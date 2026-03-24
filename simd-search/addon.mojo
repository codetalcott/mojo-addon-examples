## simd-search/addon.mojo — SIMD byte search on Node.js Buffers
##
## Three functions demonstrating byte-level SIMD impossible in pure JS:
##   1. countByte(buf, byte)      — SIMD byte counting
##   2. countLines(buf)           — count newlines (wc -l equivalent)
##   3. searchAll(buf, needle)    — all match positions → Uint32Array
##
## Build:  pixi run bash simd-search/build.sh
## Run:    node simd-search/search.js

from std.algorithm.functional import vectorize, parallelize
from std.sys import simd_width_of
from std.memory import alloc

from napi.types import NapiEnv, NapiValue
from napi.error import throw_js_error
from napi.bindings import NapiBindings, Bindings, init_bindings
from napi.framework.js_number import JsNumber
from napi.framework.js_int32 import JsInt32
from napi.framework.js_buffer import JsBuffer
from napi.framework.js_typedarray import JsTypedArray
from napi.framework.js_arraybuffer import JsArrayBuffer
from napi.framework.args import CbArgs
from napi.framework.register import fn_ptr, ModuleBuilder
from napi.framework.runtime import init_async_runtime


# --- Helper: get byte pointer + length from Buffer or Uint8Array -------------

def _get_data_ptr(b: Bindings, env: NapiEnv, val: NapiValue) raises -> UnsafePointer[Byte, MutAnyOrigin]:
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


# --- SIMD helper: count matching bytes in a SIMD vector via XOR ---------------
#
# How the XOR + bit-collapse trick works:
#   1. XOR each byte with the target. Matching bytes become 0x00, non-matches
#      become some non-zero value (at least one bit is set).
#   2. OR-shift chain: OR the byte with itself right-shifted by 1, 2, ... 7.
#      This "smears" any set bit down to the LSB. After all 7 shifts, the LSB
#      is 1 if *any* bit was originally set (non-match), 0 if the byte was 0x00.
#   3. Mask the LSB and sum: reduce_add gives the count of non-matches.
#      Subtract from width to get the match count.
#
# This avoids per-byte branching and runs entirely in SIMD registers.

def _simd_count_matches[width: Int](chunk: SIMD[DType.uint8, width], target: Byte) -> Int:
    var xored = chunk ^ SIMD[DType.uint8, width](target)
    var collapsed = xored | (xored >> 1) | (xored >> 2) | (xored >> 3) | (xored >> 4) | (xored >> 5) | (xored >> 6) | (xored >> 7)
    var non_match = collapsed & SIMD[DType.uint8, width](1)
    return width - Int(non_match.reduce_add())


# --- SIMD countByte kernel ----------------------------------------------------

comptime PARALLEL_THRESHOLD = 65536  # 64KB
comptime NUM_WORKERS = 4

def _count_byte_range(
    data: UnsafePointer[Byte, MutAnyOrigin],
    target: Byte,
    start: Int,
    end: Int,
) -> Int:
    var count: Int = 0
    var base = start
    def compute[width: Int](offset: Int) unified {mut}:
        var chunk = data.load[width=width](base + offset)
        count += _simd_count_matches(chunk, target)
    vectorize[simd_width_of[DType.uint8]()](end - start, compute)
    return count


def _count_byte(
    data: UnsafePointer[Byte, MutAnyOrigin],
    target: Byte,
    size: Int,
) -> Int:
    if size < PARALLEL_THRESHOLD:
        return _count_byte_range(data, target, 0, size)
    var chunk_size = size // NUM_WORKERS
    var partials = alloc[Int](NUM_WORKERS)
    def worker(wid: Int) capturing:
        var s = wid * chunk_size
        var e = s + chunk_size if wid < NUM_WORKERS - 1 else size
        partials[wid] = _count_byte_range(data, target, s, e)
    parallelize[worker](NUM_WORKERS)
    var total: Int = 0
    for i in range(NUM_WORKERS):
        total += partials[i]
    partials.free()
    return total


# --- SIMD position collection ------------------------------------------------

def _collect_byte_positions(
    data: UnsafePointer[Byte, MutAnyOrigin],
    target: Byte,
    size: Int,
    result: UnsafePointer[UInt32, MutAnyOrigin],
) -> Int:
    var idx: Int = 0
    comptime sw = simd_width_of[DType.uint8]()
    var full_chunks = size // sw
    for chunk_i in range(full_chunks):
        var offset = chunk_i * sw
        var chunk = data.load[width=sw](offset)
        var xored = chunk ^ SIMD[DType.uint8, sw](target)
        for lane in range(sw):
            if xored[lane] == 0:
                result[idx] = UInt32(offset + lane)
                idx += 1
    # Scalar tail
    var tail_start = full_chunks * sw
    for i in range(tail_start, size):
        if data[i] == target:
            result[idx] = UInt32(i)
            idx += 1
    return idx


# --- Multi-byte search: first+last byte SIMD filter --------------------------
#
# For multi-byte needles, checking every position against the full needle is
# expensive. Instead, we use a SIMD pre-filter:
#   1. Load a SIMD chunk at each candidate position and XOR with the first byte.
#   2. Load a second chunk offset by (needle_len - 1) and XOR with the last byte.
#   3. OR the two results: a lane is 0 only if both first AND last bytes match.
#   4. Only for those candidate lanes, verify the middle bytes with a scalar loop.
#
# This eliminates most non-matching positions in bulk via SIMD, then only does
# expensive byte-by-byte comparison on the rare candidates that pass the filter.

def _count_multi_byte(
    data: UnsafePointer[Byte, MutAnyOrigin],
    needle: UnsafePointer[Byte, MutAnyOrigin],
    data_len: Int,
    needle_len: Int,
) -> Int:
    if needle_len == 0 or needle_len > data_len:
        return 0
    var first = needle[0]
    var last = needle[needle_len - 1]
    var search_len = data_len - needle_len + 1
    var count: Int = 0
    comptime sw = simd_width_of[DType.uint8]()
    var full_chunks = search_len // sw
    for chunk_i in range(full_chunks):
        var offset = chunk_i * sw
        var first_xor = data.load[width=sw](offset) ^ SIMD[DType.uint8, sw](first)
        var last_xor = data.load[width=sw](offset + needle_len - 1) ^ SIMD[DType.uint8, sw](last)
        # candidate where both XOR == 0 (both first and last byte match)
        var combined = first_xor | last_xor  # 0 only where both match
        for lane in range(sw):
            if combined[lane] == 0:
                var pos = offset + lane
                var found = True
                for k in range(1, needle_len - 1):
                    if data[pos + k] != needle[k]:
                        found = False
                        break
                if found:
                    count += 1
    # Scalar tail
    var tail_start = full_chunks * sw
    for i in range(tail_start, search_len):
        var found = True
        for k in range(needle_len):
            if data[i + k] != needle[k]:
                found = False
                break
        if found:
            count += 1
    return count


def _collect_multi_byte(
    data: UnsafePointer[Byte, MutAnyOrigin],
    needle: UnsafePointer[Byte, MutAnyOrigin],
    data_len: Int,
    needle_len: Int,
    result: UnsafePointer[UInt32, MutAnyOrigin],
) -> Int:
    if needle_len == 0 or needle_len > data_len:
        return 0
    var first = needle[0]
    var last = needle[needle_len - 1]
    var search_len = data_len - needle_len + 1
    var idx: Int = 0
    comptime sw = simd_width_of[DType.uint8]()
    var full_chunks = search_len // sw
    for chunk_i in range(full_chunks):
        var offset = chunk_i * sw
        var first_xor = data.load[width=sw](offset) ^ SIMD[DType.uint8, sw](first)
        var last_xor = data.load[width=sw](offset + needle_len - 1) ^ SIMD[DType.uint8, sw](last)
        var combined = first_xor | last_xor
        for lane in range(sw):
            if combined[lane] == 0:
                var pos = offset + lane
                var found = True
                for k in range(1, needle_len - 1):
                    if data[pos + k] != needle[k]:
                        found = False
                        break
                if found:
                    result[idx] = UInt32(pos)
                    idx += 1
    # Scalar tail
    var tail_start = full_chunks * sw
    for i in range(tail_start, search_len):
        var found = True
        for k in range(needle_len):
            if data[i + k] != needle[k]:
                found = False
                break
        if found:
            result[idx] = UInt32(i)
            idx += 1
    return idx


# --- N-API callbacks ----------------------------------------------------------

def count_byte_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_two(env, info)
        var b = r.b
        var ptr = _get_data_ptr(b, env, r.arg0)
        var size = _get_data_len(b, env, r.arg0)
        var target = Byte(JsInt32.from_napi_value(b, env, r.arg1))
        var count = _count_byte(ptr, target, size)
        return JsNumber.create(b, env, Float64(count)).value
    except:
        throw_js_error(env, "countByte failed")
        return NapiValue()


def count_lines_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_one(env, info)
        var b = r.b
        var ptr = _get_data_ptr(b, env, r.arg0)
        var size = _get_data_len(b, env, r.arg0)
        var count = _count_byte(ptr, Byte(0x0A), size)
        return JsNumber.create(b, env, Float64(count)).value
    except:
        throw_js_error(env, "countLines failed")
        return NapiValue()


def search_all_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_two(env, info)
        var b = r.b
        var h_ptr = _get_data_ptr(b, env, r.arg0)
        var h_len = _get_data_len(b, env, r.arg0)
        var n_ptr = _get_data_ptr(b, env, r.arg1)
        var n_len = _get_data_len(b, env, r.arg1)

        # Count matches first to allocate exact-size result
        var match_count: Int
        if n_len == 1:
            match_count = _count_byte(h_ptr, n_ptr[0], h_len)
        else:
            match_count = _count_multi_byte(h_ptr, n_ptr, h_len, n_len)

        # Create result Uint32Array
        var byte_len = UInt(match_count * 4) if match_count > 0 else UInt(0)
        var ab = JsArrayBuffer.create(b, env, byte_len)
        if match_count > 0:
            var ab_ptr = ab.data_ptr(b, env).bitcast[UInt32]()
            if n_len == 1:
                _ = _collect_byte_positions(h_ptr, n_ptr[0], h_len, ab_ptr)
            else:
                _ = _collect_multi_byte(h_ptr, n_ptr, h_len, n_len, ab_ptr)

        return JsTypedArray.create_uint32(b, env, ab.value, 0, UInt(match_count)).value
    except:
        throw_js_error(env, "searchAll failed")
        return NapiValue()


# --- Module entry point -------------------------------------------------------

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

    var cb_ref = count_byte_fn
    var cl_ref = count_lines_fn
    var sa_ref = search_all_fn

    try:
        var m = ModuleBuilder(env, exports, cb_data)
        m.method("countByte", fn_ptr(cb_ref))
        m.method("countLines", fn_ptr(cl_ref))
        m.method("searchAll", fn_ptr(sa_ref))
        m.flush()
    except:
        pass

    return exports
