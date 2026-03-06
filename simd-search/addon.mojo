## simd-search/addon.mojo — SIMD byte search on Node.js Buffers
##
## Three functions demonstrating byte-level SIMD impossible in pure JS:
##   1. countByte(buf, byte)      — SIMD byte counting
##   2. countLines(buf)           — count newlines (wc -l equivalent)
##   3. searchAll(buf, needle)    — all match positions → Uint32Array
##
## Build:  pixi run bash simd-search/build.sh
## Run:    node simd-search/search.js

from algorithm.functional import vectorize, parallelize
from sys import simd_width_of
from memory import alloc
from ffi import OwnedDLHandle

from napi.types import NapiEnv, NapiValue, NapiStatus, NAPI_UINT32_ARRAY
from napi.error import throw_js_error, check_status
from napi.framework.js_number import JsNumber
from napi.framework.js_int32 import JsInt32
from napi.framework.js_buffer import JsBuffer
from napi.framework.js_typedarray import JsTypedArray
from napi.framework.js_arraybuffer import JsArrayBuffer
from napi.framework.args import CbArgs
from napi.framework.register import fn_ptr, ModuleBuilder
from napi.raw import raw_create_typedarray


# --- Mojo async runtime init -------------------------------------------------

fn _init_mojo_async_runtime() raises:
    var lib = OwnedDLHandle()
    var create_rt = lib.get_function[
        fn () -> OpaquePointer[MutAnyOrigin]
    ]("KGEN_CompilerRT_AsyncRT_CreateRuntime")
    _ = create_rt()


# --- Helper: get byte pointer + length from Buffer or Uint8Array -------------

fn _get_data_ptr(env: NapiEnv, val: NapiValue) raises -> UnsafePointer[Byte, MutAnyOrigin]:
    if JsBuffer.is_buffer(env, val):
        return JsBuffer(val).data_ptr(env)
    if JsTypedArray.is_typedarray(env, val):
        return JsTypedArray(val).data_ptr(env)
    raise Error("expected Buffer or Uint8Array")

fn _get_data_len(env: NapiEnv, val: NapiValue) raises -> Int:
    if JsBuffer.is_buffer(env, val):
        return Int(JsBuffer(val).length(env))
    if JsTypedArray.is_typedarray(env, val):
        return Int(JsTypedArray(val).length(env))
    raise Error("expected Buffer or Uint8Array")


# --- SIMD helper: count matching bytes in a SIMD vector via XOR ---------------
# XOR with target splat: matching bytes become 0x00.
# Collapse all bits per byte with OR chain, mask bit 0: 0 = match, 1 = non-match.
# Count = width - reduce_add(non_match_bits).

fn _simd_count_matches[width: Int](chunk: SIMD[DType.uint8, width], target: Byte) -> Int:
    var xored = chunk ^ SIMD[DType.uint8, width](target)
    # Collapse each byte to 1 if any bit set (non-match), 0 if all zero (match)
    var collapsed = xored | (xored >> 1) | (xored >> 2) | (xored >> 3) | (xored >> 4) | (xored >> 5) | (xored >> 6) | (xored >> 7)
    var non_match = collapsed & SIMD[DType.uint8, width](1)
    return width - Int(non_match.reduce_add())


# --- SIMD countByte kernel ----------------------------------------------------

comptime PARALLEL_THRESHOLD = 65536  # 64KB
comptime NUM_WORKERS = 4

fn _count_byte_range(
    data: UnsafePointer[Byte, MutAnyOrigin],
    target: Byte,
    start: Int,
    end: Int,
) -> Int:
    var count: Int = 0
    var base = start
    fn compute[width: Int](offset: Int) unified {mut}:
        var chunk = data.load[width=width](base + offset)
        count += _simd_count_matches(chunk, target)
    vectorize[simd_width_of[DType.uint8]()](end - start, compute)
    return count


fn _count_byte(
    data: UnsafePointer[Byte, MutAnyOrigin],
    target: Byte,
    size: Int,
) -> Int:
    if size < PARALLEL_THRESHOLD:
        return _count_byte_range(data, target, 0, size)
    var chunk_size = size // NUM_WORKERS
    var partials = alloc[Int](NUM_WORKERS)
    fn worker(wid: Int) capturing:
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

fn _collect_byte_positions(
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


# --- Multi-byte search: first+last byte SIMD technique -----------------------

fn _count_multi_byte(
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


fn _collect_multi_byte(
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

fn count_byte_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var args = CbArgs.get_two(env, info)
        var ptr = _get_data_ptr(env, args[0])
        var size = _get_data_len(env, args[0])
        var target = Byte(JsInt32.from_napi_value(env, args[1]))
        var count = _count_byte(ptr, target, size)
        return JsNumber.create(env, Float64(count)).value
    except:
        throw_js_error(env, "countByte failed")
        return NapiValue()


fn count_lines_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var arg = CbArgs.get_one(env, info)
        var ptr = _get_data_ptr(env, arg)
        var size = _get_data_len(env, arg)
        var count = _count_byte(ptr, Byte(0x0A), size)
        return JsNumber.create(env, Float64(count)).value
    except:
        throw_js_error(env, "countLines failed")
        return NapiValue()


fn search_all_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var args = CbArgs.get_two(env, info)
        var h_ptr = _get_data_ptr(env, args[0])
        var h_len = _get_data_len(env, args[0])
        var n_ptr = _get_data_ptr(env, args[1])
        var n_len = _get_data_len(env, args[1])

        # Count matches first to allocate exact-size result
        var match_count: Int
        if n_len == 1:
            match_count = _count_byte(h_ptr, n_ptr[0], h_len)
        else:
            match_count = _count_multi_byte(h_ptr, n_ptr, h_len, n_len)

        # Create result Uint32Array
        var byte_len = UInt(match_count * 4) if match_count > 0 else UInt(0)
        var ab = JsArrayBuffer.create(env, byte_len)
        if match_count > 0:
            var ab_ptr = ab.data_ptr(env).bitcast[UInt32]()
            if n_len == 1:
                _ = _collect_byte_positions(h_ptr, n_ptr[0], h_len, ab_ptr)
            else:
                _ = _collect_multi_byte(h_ptr, n_ptr, h_len, n_len, ab_ptr)

        var result = NapiValue()
        check_status(raw_create_typedarray(env, NAPI_UINT32_ARRAY, UInt(match_count),
            ab.value, 0, UnsafePointer(to=result).bitcast[NoneType]()))
        return result
    except:
        throw_js_error(env, "searchAll failed")
        return NapiValue()


# --- Module entry point -------------------------------------------------------

@export("napi_register_module_v1", ABI="C")
fn register_module(env: NapiEnv, exports: NapiValue) -> NapiValue:
    try:
        _init_mojo_async_runtime()
    except:
        pass

    var cb_ref = count_byte_fn
    var cl_ref = count_lines_fn
    var sa_ref = search_all_fn

    try:
        var m = ModuleBuilder(env, exports)
        m.method("countByte", fn_ptr(cb_ref))
        m.method("countLines", fn_ptr(cl_ref))
        m.method("searchAll", fn_ptr(sa_ref))
    except:
        pass

    return exports
