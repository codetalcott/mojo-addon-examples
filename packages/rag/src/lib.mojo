## packages/rag/src/lib.mojo — @qkstat/rag N-API entry point
##
## Compiled into build/rag.node. End users load it via `require('@qkstat/rag')`,
## which resolves to a platform-specific prebuilt sub-package. Registers the
## four RAG primitives: loadMatrixGpu, matmulHandle, searchHandle,
## releaseMatrixGpu (see src/linalg.mojo).
##
## Builds against napi-mojo's N-API framework via build.sh's
## `-I node_modules/napi-mojo/src` include.

from std.memory import alloc
from napi.types import NapiEnv, NapiValue
from napi.bindings import NapiBindings, init_bindings
from napi.raw import raw_create_error, raw_fatal_exception
from napi.framework.js_string import JsString
from napi.framework.register import ModuleBuilder
from kernels import register_gpu_linalg


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
        register_gpu_linalg(m, bindings_ptr.as_unsafe_any_origin())
        m.flush()
    except:
        try:
            var null_code = NapiValue(unsafe_from_address=Int(0))
            var err_msg = JsString.create_literal(
                env, "@qkstat/rag: register_module failed"
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
