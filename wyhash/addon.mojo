## wyhash/addon.mojo — wyhash non-cryptographic hash on Buffers
##
## Two functions: match C hash performance in ~50 lines of Mojo.
##   1. wyHash(buf, seed?)    → BigInt (full 64-bit hash)
##   2. wyHash64(buf, seed?)  → Number (lossy Float64, no BigInt overhead)
##
## Build:  pixi run bash wyhash/build.sh
## Run:    node wyhash/hash.js

from std.memory import bitcast, alloc

from napi.types import NapiEnv, NapiValue, NAPI_TYPE_NUMBER, NAPI_TYPE_BIGINT
from napi.error import throw_js_error
from napi.bindings import NapiBindings, Bindings, init_bindings
from napi.framework.js_number import JsNumber
from napi.framework.js_bigint import JsBigInt
from napi.framework.js_buffer import JsBuffer
from napi.framework.js_typedarray import JsTypedArray
from napi.framework.js_value import js_typeof
from napi.framework.args import CbArgs
from napi.framework.register import fn_ptr, ModuleBuilder


# --- wyhash v4.2 constants ---------------------------------------------------

comptime _WYP0: UInt64 = 0xa0761d6478bd642f
comptime _WYP1: UInt64 = 0xe7037ed1a0b428db
comptime _WYP2: UInt64 = 0x8ebc6af09c88c6e3
comptime _WYP3: UInt64 = 0x589965cc75374cc3


# --- Core primitives ---------------------------------------------------------

def _wymum(a: UInt64, b: UInt64) -> UInt64:
    """128-bit folded multiply: (a * b) as 128-bit, return lo XOR hi."""
    var m = a.cast[DType.uint128]() * b.cast[DType.uint128]()
    var parts = bitcast[DType.uint64, 2](m)
    return parts[0] ^ parts[1]


def _wyr8(p: UnsafePointer[Byte, MutAnyOrigin], offset: Int) -> UInt64:
    """Read 8 bytes as little-endian UInt64."""
    return (p + offset).bitcast[UInt64]()[]


def _wyr4(p: UnsafePointer[Byte, MutAnyOrigin], offset: Int) -> UInt64:
    """Read 4 bytes as little-endian UInt32, zero-extend to UInt64."""
    return UInt64((p + offset).bitcast[UInt32]()[])


def _wyr3(p: UnsafePointer[Byte, MutAnyOrigin], k: Int, length: Int) -> UInt64:
    """Read 1-3 bytes into a UInt64."""
    return (UInt64(p[k]) << 16) | (UInt64(p[k + (length >> 1)]) << 8) | UInt64(p[k + length - 1])


# --- wyhash main function ----------------------------------------------------

def wyhash(data: UnsafePointer[Byte, MutAnyOrigin], length: Int, in_seed: UInt64) -> UInt64:
    var seed = in_seed ^ _wymum(in_seed ^ _WYP0, _WYP1)
    var a: UInt64 = 0
    var b: UInt64 = 0

    if length <= 16:
        if length >= 4:
            a = (_wyr4(data, 0) << 32) | _wyr4(data, (length >> 3) << 2)
            b = (_wyr4(data, length - 4) << 32) | _wyr4(data, length - ((length >> 3) << 2) - 4)
        elif length > 0:
            a = _wyr3(data, 0, length)
            b = 0
    elif length <= 48:
        # 17-48 bytes: first 16 always safe, second 16 only if len > 32
        seed = _wymum(_wyr8(data, 0) ^ _WYP1, _wyr8(data, 8) ^ seed)
        if length > 32:
            seed = _wymum(_wyr8(data, 16) ^ _WYP2, _wyr8(data, 24) ^ seed)
        a = _wyr8(data, length - 16)
        b = _wyr8(data, length - 8)
    else:
        # Bulk: 48-byte chunks, 3-way seed mixing
        var see1 = seed
        var see2 = seed
        var i = 0
        var remaining = length
        while remaining > 48:
            seed = _wymum(_wyr8(data, i) ^ _WYP1, _wyr8(data, i + 8) ^ seed)
            see1 = _wymum(_wyr8(data, i + 16) ^ _WYP2, _wyr8(data, i + 24) ^ see1)
            see2 = _wymum(_wyr8(data, i + 32) ^ _WYP3, _wyr8(data, i + 40) ^ see2)
            i += 48
            remaining -= 48
        seed ^= see1 ^ see2
        # Process remaining 1-48 bytes (re-read from end)
        var tail = length - remaining
        if remaining > 32:
            seed = _wymum(_wyr8(data, tail) ^ _WYP1, _wyr8(data, tail + 8) ^ seed)
            see1 = _wymum(_wyr8(data, tail + 16) ^ _WYP2, _wyr8(data, tail + 24) ^ see1)
            seed ^= see1
        elif remaining > 16:
            seed = _wymum(_wyr8(data, tail) ^ _WYP1, _wyr8(data, tail + 8) ^ seed)
        a = _wyr8(data, length - 16)
        b = _wyr8(data, length - 8)

    return _wymum(_WYP1 ^ UInt64(length), _wymum(a ^ _WYP1, b ^ seed))


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


# --- Read seed argument (Number or BigInt, def ault 0) -------------------------

def _read_seed(b: Bindings, env: NapiEnv, val: NapiValue) raises -> UInt64:
    var t = js_typeof(b, env, val)
    if t == NAPI_TYPE_NUMBER:
        return UInt64(Int64(JsNumber.from_napi_value(b, env, val)))
    if t == NAPI_TYPE_BIGINT:
        return JsBigInt.to_uint64(b, env, val)
    return UInt64(0)


# --- N-API callbacks ----------------------------------------------------------

def wy_hash_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    """wyHash(buf, seed?) → BigInt"""
    try:
        var bindings = CbArgs.get_bindings(env, info)
        var argc = CbArgs.argc(bindings, env, info)
        var arg0 = CbArgs.get_one(bindings, env, info)
        var ptr = _get_data_ptr(bindings, env, arg0)
        var length = _get_data_len(bindings, env, arg0)
        var seed = UInt64(0)
        if argc >= 2:
            var args = CbArgs.get_two(bindings, env, info)
            seed = _read_seed(bindings, env, args[1])
        var result = wyhash(ptr, length, seed)
        return JsBigInt.from_uint64(bindings, env, result).value
    except:
        throw_js_error(env, "wyHash failed")
        return NapiValue()


def wy_hash64_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    """wyHash64(buf, seed?) → Number (lossy Float64)"""
    try:
        var bindings = CbArgs.get_bindings(env, info)
        var argc = CbArgs.argc(bindings, env, info)
        var arg0 = CbArgs.get_one(bindings, env, info)
        var ptr = _get_data_ptr(bindings, env, arg0)
        var length = _get_data_len(bindings, env, arg0)
        var seed = UInt64(0)
        if argc >= 2:
            var args = CbArgs.get_two(bindings, env, info)
            seed = _read_seed(bindings, env, args[1])
        var result = wyhash(ptr, length, seed)
        return JsNumber.create(bindings, env, Float64(result)).value
    except:
        throw_js_error(env, "wyHash64 failed")
        return NapiValue()


# --- Module entry point -------------------------------------------------------

@export("napi_register_module_v1", ABI="C")
def register_module(env: NapiEnv, exports: NapiValue) -> NapiValue:
    var bindings_ptr = alloc[NapiBindings](1)
    try:
        var bindings = NapiBindings()
        init_bindings(bindings)
        bindings_ptr.init_pointee_move(bindings^)
    except:
        bindings_ptr.free()
        return exports
    var cb_data = bindings_ptr.bitcast[NoneType]()

    var wh_ref = wy_hash_fn
    var wh64_ref = wy_hash64_fn

    try:
        var m = ModuleBuilder(env, exports, cb_data)
        m.method("wyHash", fn_ptr(wh_ref))
        m.method("wyHash64", fn_ptr(wh64_ref))
        m.flush()
    except:
        pass

    return exports
