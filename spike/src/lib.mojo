## spike/src/lib.mojo — embedding-kernel spike N-API entry point
##
## Compiles into spike/build/embed.node. Load from Node via
## `require('./spike/build/embed.node')`.
##
## Registers one export for the Day 1 smoke test:
##   embedTokens(tokenIds: Int32Array, attentionMask: Int32Array,
##               batch: Int, seqLen: Int, dstEmbeddings: Float32Array) -> 0
##
## Day 1: stub fills dst with zeros, proves the N-API boundary compiles.
## Day 2+: wires to MAX graph in embed.mojo.

from std.memory import alloc
from napi.types import NapiEnv, NapiValue
from napi.bindings import NapiBindings, init_bindings
from napi.raw import raw_create_error, raw_fatal_exception
from napi.framework.js_string import JsString
from napi.framework.register import ModuleBuilder
from embed import register_embed


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

    try:
        var m = ModuleBuilder(env, exports, cb_data)
        register_embed(m, bindings_ptr)
        m.flush()
    except:
        try:
            var null_code = NapiValue()
            var err_msg = JsString.create_literal(
                env, "spike/embed: register_module failed"
            )
            var err_val = NapiValue()
            var err_ptr: OpaquePointer[MutAnyOrigin] = UnsafePointer(
                to=err_val
            ).bitcast[NoneType]()
            _ = raw_create_error(env, null_code, err_msg.value, err_ptr)
            _ = raw_fatal_exception(env, err_val)
        except:
            pass

    return exports
