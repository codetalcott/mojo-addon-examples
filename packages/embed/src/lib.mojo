## packages/embed/src/lib.mojo — @qkstat/embed N-API entry point
##
## Compiles into packages/embed/build/embed.node. Load from Node via
## `require('@qkstat/embed')` (or `require('./packages/embed/build/embed.node')`
## during local development).
##
## Exports:
##   embedTokens(tokenIds: Int32Array, attentionMask: Int32Array,
##               batch: Int, seqLen: Int, dstEmbeddings: Float32Array) -> 0

from std.memory import alloc
from napi.types import NapiEnv, NapiValue
from napi.bindings import NapiBindings, init_bindings
from napi.raw import raw_create_error, raw_fatal_exception
from napi.framework.js_string import JsString
from napi.framework.register import ModuleBuilder
from embed import register_embed


@export("napi_register_module_v1")
def register_module(env: NapiEnv, exports: NapiValue) abi("C") -> NapiValue:
    var bindings_ptr = alloc[NapiBindings](1)
    try:
        var bindings = NapiBindings()
        init_bindings(bindings)
        bindings_ptr.unsafe_write(bindings^)
    except:
        bindings_ptr.free()
        return exports
    var cb_data = bindings_ptr.bitcast[NoneType]().as_unsafe_any_origin()

    try:
        var m = ModuleBuilder(env, exports, cb_data)
        register_embed(m, bindings_ptr.as_unsafe_any_origin())
        m.flush()
    except:
        try:
            var null_code = NapiValue(unsafe_from_address=Int(0))
            var err_msg = JsString.create_literal(
                env, "@qkstat/embed: register_module failed"
            )
            var err_val = NapiValue(unsafe_from_address=Int(0))
            var err_ptr: OpaquePointer[MutAnyOrigin] = UnsafePointer(
                to=err_val
            ).bitcast[NoneType]().as_unsafe_any_origin()
            _ = raw_create_error(env, null_code, err_msg.value, err_ptr)
            _ = raw_fatal_exception(env, err_val)
        except:
            pass

    return exports
