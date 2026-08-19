## packages/retrieve/src/lib.mojo — @qkstat/retrieve N-API entry point
##
## Compiled into build/retrieve.node. End users load it via `require('@qkstat/retrieve')`,
## which resolves to a platform-specific prebuilt sub-package. Registers the
## four RAG primitives: loadMatrixGpu, matmulHandle, searchHandle,
## releaseMatrixGpu (see src/linalg.mojo).
##
## Builds against napi-mojo's N-API framework via build.sh's
## `-I node_modules/napi-mojo/src` include.

from std.memory.alloc import unsafe_alloc
from napi.types import NapiEnv, NapiValue
from napi.bindings import NapiBindings, init_bindings
from napi.error import throw_js_error
from napi.framework.register import ModuleBuilder
from kernels import register_gpu_linalg


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
        register_gpu_linalg(m, bindings_ptr.as_unsafe_any_origin())
        m.flush()
    except:
        # Leaves a pending JS error so require() throws with a real message.
        # throw_js_error is env-only by design — it is the fallback for
        # contexts where cached bindings could not be obtained.
        throw_js_error(env, "@qkstat/retrieve: register_module failed")

    return exports
