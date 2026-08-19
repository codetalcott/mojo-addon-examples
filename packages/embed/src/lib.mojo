## packages/embed/src/lib.mojo — @qkstat/embed N-API entry point
##
## Compiles into packages/embed/build/embed.node. Load from Node via
## `require('@qkstat/embed')` (or `require('./packages/embed/build/embed.node')`
## during local development).
##
## Exports:
##   embedTokens(tokenIds: Int32Array, attentionMask: Int32Array,
##               batch: Int, seqLen: Int, dstEmbeddings: Float32Array) -> 0

from std.memory.alloc import unsafe_alloc
from napi.types import NapiEnv, NapiValue
from napi.bindings import NapiBindings, init_bindings
from napi.error import throw_js_error
from napi.framework.register import ModuleBuilder
from embed import register_embed


@export("napi_register_module_v1")
def register_module(env: NapiEnv, exports: NapiValue) abi("C") -> NapiValue:
    var bindings_ptr = unsafe_alloc[NapiBindings](1)
    try:
        var bindings = NapiBindings()
        init_bindings(bindings)
        bindings_ptr.unsafe_write(bindings^)
    except:
        bindings_ptr.unsafe_free()
        return exports
    var cb_data = bindings_ptr.unsafe_bitcast[NoneType]().as_unsafe_any_origin()

    try:
        var m = ModuleBuilder(env, exports, cb_data)
        register_embed(m, bindings_ptr.as_unsafe_any_origin())
        m.flush()
    except:
        # Leaves a pending JS error so require() throws with a real message.
        # throw_js_error is env-only by design — it is the fallback for
        # contexts where cached bindings could not be obtained.
        throw_js_error(env, "@qkstat/embed: register_module failed")

    return exports
