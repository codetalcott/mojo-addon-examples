## matmul/addon_cached.mojo — persistent device buffer + linalg.matmul GPU
##
## Phase 3c.2: upload matrices A and B to the GPU once via loadMatrixGpu,
## then run production-grade GPU matmul (C = A × B) many times without
## per-call H2D transfer or device allocation. Uses MAX's linalg.matmul
## which dispatches to tensor cores on NVIDIA (TF32 for FP32 inputs on
## H100, FP16 tensor cores for FP16) and Metal on Apple silicon.
##
## API (as a separate .node from matmul.node):
##   loadMatrixGpu(f32Array, rows, cols)  -> External handle
##   matmulHandle(hA, hB, dstF32Array)    -> fills dst in place (C = A × B)
##   releaseMatrixGpu(h)                  -> tombstone; memory freed on GC
##
## Build: pixi run bash matmul/build_cached.sh

from std.math import ceildiv
from std.memory import alloc, memcpy
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul import matmul as linalg_matmul

from napi.types import NapiEnv, NapiValue
from napi.error import throw_js_error
from napi.bindings import NapiBindings, Bindings, init_bindings
from napi.framework.instance_data import set_instance_data, get_instance_data
from napi.framework.js_number import JsNumber
from napi.framework.js_int32 import JsInt32
from napi.framework.js_typedarray import JsTypedArray
from napi.framework.js_external import JsExternal
from napi.framework.args import CbArgs
from napi.framework.register import fn_ptr, ModuleBuilder
from napi.framework.runtime import init_async_runtime


comptime dtype = DType.float32


# --- GpuState ----------------------------------------------------------------

struct GpuState(Movable):
    var ctx: DeviceContext

    def __init__(out self, var ctx: DeviceContext):
        self.ctx = ctx^


def _get_gpu_state(
    b: Bindings, env: NapiEnv
) raises -> UnsafePointer[GpuState, MutAnyOrigin]:
    try:
        return get_instance_data[GpuState](b, env)
    except:
        raise Error("loadMatrixGpu requires a GPU (no accelerator found)")


# --- CachedMatrix: persistent device buffer for one matrix ------------------

struct CachedMatrix(Movable):
    var dev_data: DeviceBuffer[DType.float32]
    var rows: Int
    var cols: Int
    var released: Bool

    def __init__(
        out self,
        var dev_data: DeviceBuffer[DType.float32],
        rows: Int,
        cols: Int,
    ):
        self.dev_data = dev_data^
        self.rows = rows
        self.cols = cols
        self.released = False


# --- loadMatrixGpu: one-shot H2D upload -------------------------------------

def _load_matrix_gpu(
    ctx: DeviceContext,
    src_bytes: UnsafePointer[Byte, MutAnyOrigin],
    rows: Int,
    cols: Int,
) raises -> CachedMatrix:
    var num_elems = rows * cols
    var num_bytes = num_elems * 4

    var dev_data = ctx.enqueue_create_buffer[dtype](num_elems)
    var staging = ctx.enqueue_create_host_buffer[dtype](num_elems)
    memcpy(
        dest=staging.unsafe_ptr().bitcast[Byte](),
        src=src_bytes,
        count=num_bytes,
    )
    ctx.enqueue_copy(dev_data, staging)
    ctx.synchronize()

    return CachedMatrix(dev_data^, rows, cols)


def load_matrix_gpu_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var b = CbArgs.get_bindings(env, info)
        var args = CbArgs.get_three(b, env, info)
        var ta = JsTypedArray(args[0])
        var rows = Int(JsInt32.from_napi_value(b, env, args[1]))
        var cols = Int(JsInt32.from_napi_value(b, env, args[2]))
        var src_ptr = ta.data_ptr(b, env)
        var state = _get_gpu_state(b, env)

        var cm_val = _load_matrix_gpu(state[].ctx, src_ptr, rows, cols)
        return JsExternal.create_typed(b, env, cm_val^).value
    except:
        throw_js_error(env, "loadMatrixGpu failed (no GPU or upload error)")
        return NapiValue()


# --- matmulHandle: C = A × B using linalg.matmul ----------------------------

def _matmul_cached(
    ctx: DeviceContext,
    a: UnsafePointer[CachedMatrix, MutAnyOrigin],
    b: UnsafePointer[CachedMatrix, MutAnyOrigin],
    dst_bytes: UnsafePointer[Byte, MutAnyOrigin],
) raises:
    var M = a[].rows
    var K = a[].cols
    var N = b[].cols
    var c_elems = M * N

    # Per-call C buffer (device + pinned host for D2H).
    var dev_c = ctx.enqueue_create_buffer[dtype](c_elems)
    dev_c.enqueue_fill(0.0)
    var host_c = ctx.enqueue_create_host_buffer[dtype](c_elems)

    # Wrap persistent A, B and per-call C as TileTensors.
    var tt_a = TileTensor[dtype](a[].dev_data, row_major(Coord(Idx(M), Idx(K))))
    var tt_b = TileTensor[dtype](b[].dev_data, row_major(Coord(Idx(K), Idx(N))))
    var tt_c = TileTensor[dtype](dev_c, row_major(Coord(Idx(M), Idx(N))))

    linalg_matmul[target="gpu"](tt_c, tt_a, tt_b, Optional(ctx))

    ctx.enqueue_copy(host_c, dev_c)
    ctx.synchronize()

    memcpy(
        dest=dst_bytes,
        src=host_c.unsafe_ptr().bitcast[Byte](),
        count=c_elems * 4,
    )


def matmul_handle_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var b = CbArgs.get_bindings(env, info)
        var args = CbArgs.get_three(b, env, info)

        var a = JsExternal.get_typed[CachedMatrix](
            b, env, args[0], "matmulHandle A"
        )
        if a[].released:
            raise Error("matmulHandle: handle A has been released")

        var b_mat = JsExternal.get_typed[CachedMatrix](
            b, env, args[1], "matmulHandle B"
        )
        if b_mat[].released:
            raise Error("matmulHandle: handle B has been released")

        # Dimension check: A.cols must equal B.rows
        if a[].cols != b_mat[].rows:
            raise Error("matmulHandle: dimension mismatch (A.cols != B.rows)")

        # arg 2: caller-provided Float32Array destination for C
        var dst_ta = JsTypedArray(args[2])
        var dst_len = Int(dst_ta.length(b, env))
        var expected_len = a[].rows * b_mat[].cols
        if dst_len < expected_len:
            raise Error("matmulHandle: dst buffer too small")
        var dst_ptr = dst_ta.data_ptr(b, env)

        var state = _get_gpu_state(b, env)
        _matmul_cached(state[].ctx, a, b_mat, dst_ptr)
        return JsNumber.create(b, env, 0.0).value
    except:
        throw_js_error(env, "matmulHandle failed")
        return NapiValue()


# --- releaseMatrixGpu --------------------------------------------------------

def release_matrix_gpu_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_one(env, info)
        var b = r.b
        var cm = JsExternal.get_typed[CachedMatrix](
            b, env, r.arg0, "releaseMatrixGpu"
        )
        cm[].released = True
        return JsNumber.create(b, env, 0.0).value
    except:
        throw_js_error(env, "releaseMatrixGpu failed")
        return NapiValue()


# --- Module entry point ------------------------------------------------------

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

    try:
        var ctx = DeviceContext()
        set_instance_data(bindings_ptr, env, GpuState(ctx^))
    except:
        pass

    var lmg_ref = load_matrix_gpu_fn
    var mh_ref = matmul_handle_fn
    var rmg_ref = release_matrix_gpu_fn

    try:
        var m = ModuleBuilder(env, exports, cb_data)
        m.method("loadMatrixGpu", fn_ptr(lmg_ref))
        m.method("matmulHandle", fn_ptr(mh_ref))
        m.method("releaseMatrixGpu", fn_ptr(rmg_ref))
        m.flush()
    except:
        pass

    return exports
