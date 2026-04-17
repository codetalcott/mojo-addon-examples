## matmul/addon.mojo — Progressive matrix multiply optimization
##
## Four implementations showing Mojo's optimization story:
##   1. matmulNaive      — triple loop baseline
##   2. matmulVectorized  — SIMD vectorize() inner loop
##   3. matmulTiled       — cache-friendly blocking
##   4. matmulParallel    — parallelize() across tile rows
##
## All operate on row-major Float64Arrays: C[i,j] = sum_k A[i,k] * B[k,j]
## JS passes pre-allocated output buffer for zero-allocation benchmarking.
##
## Build:  pixi run bash matmul/build.sh
## Run:    node matmul/matmul.js

from std.algorithm.functional import vectorize, parallelize
from std.sys import simd_width_of
from std.memory import alloc

from napi.types import NapiEnv, NapiValue
from napi.error import throw_js_error
from napi.bindings import NapiBindings, Bindings, init_bindings
from napi.framework.js_number import JsNumber
from napi.framework.js_int32 import JsInt32
from napi.framework.js_typedarray import JsTypedArray
from napi.framework.args import CbArgs
from napi.framework.register import fn_ptr, ModuleBuilder
from napi.framework.runtime import init_async_runtime


# --- Helper: extract matmul args from JS -------------------------------------
# Args: (a: Float64Array, b: Float64Array, result: Float64Array, M: int, K: int, N: int)

def parse_matmul_args(b: Bindings, env: NapiEnv, info: NapiValue) raises -> InlineArray[
    UnsafePointer[Float64, MutAnyOrigin], 3
]:
    var argc = CbArgs.argc(b, env, info)
    if argc < 6:
        raise Error("matmul requires 6 arguments: a, b, result, M, K, N")
    var argv_buf = alloc[NapiValue](6)
    CbArgs.get_argv(b, env, info, 6, argv_buf)
    var ta_a = JsTypedArray(argv_buf[0])
    var ta_b = JsTypedArray(argv_buf[1])
    var ta_out = JsTypedArray(argv_buf[2])
    var ptr_a = ta_a.data_ptr(b, env).bitcast[Float64]()
    var ptr_b = ta_b.data_ptr(b, env).bitcast[Float64]()
    var ptr_out = ta_out.data_ptr(b, env).bitcast[Float64]()
    argv_buf.free()
    var ptrs = InlineArray[UnsafePointer[Float64, MutAnyOrigin], 3](fill=ptr_a)
    ptrs[1] = ptr_b
    ptrs[2] = ptr_out
    return ptrs^

def parse_dims(b: Bindings, env: NapiEnv, info: NapiValue) raises -> InlineArray[Int, 3]:
    var argv_buf = alloc[NapiValue](6)
    CbArgs.get_argv(b, env, info, 6, argv_buf)
    var M = Int(JsInt32.from_napi_value(b, env, argv_buf[3]))
    var K = Int(JsInt32.from_napi_value(b, env, argv_buf[4]))
    var N = Int(JsInt32.from_napi_value(b, env, argv_buf[5]))
    argv_buf.free()
    var dims = InlineArray[Int, 3](fill=M)
    dims[1] = K
    dims[2] = N
    return dims^


# --- 1. Naive: triple loop ---------------------------------------------------

def _matmul_naive(
    a: UnsafePointer[Float64, MutAnyOrigin],
    b: UnsafePointer[Float64, MutAnyOrigin],
    c: UnsafePointer[Float64, MutAnyOrigin],
    M: Int, K: Int, N: Int,
):
    for i in range(M):
        for j in range(N):
            var sum: Float64 = 0.0
            for p in range(K):
                sum += a[i * K + p] * b[p * N + j]
            c[i * N + j] = sum


def matmul_naive_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var bindings = CbArgs.get_bindings(env, info)
        var ptrs = parse_matmul_args(bindings, env, info)
        var dims = parse_dims(bindings, env, info)
        _matmul_naive(ptrs[0], ptrs[1], ptrs[2], dims[0], dims[1], dims[2])
        return JsNumber.create(bindings, env, 0.0).value
    except:
        throw_js_error(env, "matmulNaive failed")
        return NapiValue()


# --- 2. Vectorized: SIMD inner loop ------------------------------------------
# Reorder to i,k,j (row-of-B access pattern) so the inner j-loop is contiguous
# and can be vectorized. Each SIMD lane processes multiple j values at once.

def _matmul_vectorized(
    a: UnsafePointer[Float64, MutAnyOrigin],
    b: UnsafePointer[Float64, MutAnyOrigin],
    c: UnsafePointer[Float64, MutAnyOrigin],
    M: Int, K: Int, N: Int,
):
    for i in range(M * N):
        c[i] = 0.0
    for i in range(M):
        var row_c = i * N
        for p in range(K):
            var a_ip = a[i * K + p]
            var row_b = p * N
            def compute[width: Int](j: Int) unified {read}:
                var b_chunk = b.load[width=width](row_b + j)
                var c_chunk = c.load[width=width](row_c + j)
                c.store[width=width](row_c + j, c_chunk + a_ip * b_chunk)
            vectorize[simd_width_of[DType.float64]()](N, compute)


def matmul_vectorized_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var bindings = CbArgs.get_bindings(env, info)
        var ptrs = parse_matmul_args(bindings, env, info)
        var dims = parse_dims(bindings, env, info)
        _matmul_vectorized(ptrs[0], ptrs[1], ptrs[2], dims[0], dims[1], dims[2])
        return JsNumber.create(bindings, env, 0.0).value
    except:
        throw_js_error(env, "matmulVectorized failed")
        return NapiValue()


# --- 3. Tiled: cache-friendly blocking ----------------------------------------
# Blocks the computation into TILE_SIZE x TILE_SIZE tiles that fit in L1/L2.
# Within each tile, uses SIMD vectorize on the inner j-loop.

comptime TILE_SIZE = 64

def _matmul_tiled(
    a: UnsafePointer[Float64, MutAnyOrigin],
    b: UnsafePointer[Float64, MutAnyOrigin],
    c: UnsafePointer[Float64, MutAnyOrigin],
    M: Int, K: Int, N: Int,
):
    for i in range(M * N):
        c[i] = 0.0
    var ii = 0
    while ii < M:
        var i_end = ii + TILE_SIZE if ii + TILE_SIZE < M else M
        var pp = 0
        while pp < K:
            var p_end = pp + TILE_SIZE if pp + TILE_SIZE < K else K
            var jj = 0
            while jj < N:
                var j_end = jj + TILE_SIZE if jj + TILE_SIZE < N else N
                var tile_n = j_end - jj
                for i in range(ii, i_end):
                    var row_c = i * N + jj
                    for p in range(pp, p_end):
                        var a_ip = a[i * K + p]
                        var row_b = p * N + jj
                        def compute[width: Int](j: Int) unified {read}:
                            var b_chunk = b.load[width=width](row_b + j)
                            var c_chunk = c.load[width=width](row_c + j)
                            c.store[width=width](row_c + j, c_chunk + a_ip * b_chunk)
                        vectorize[simd_width_of[DType.float64]()](tile_n, compute)
                jj += TILE_SIZE
            pp += TILE_SIZE
        ii += TILE_SIZE


def matmul_tiled_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var bindings = CbArgs.get_bindings(env, info)
        var ptrs = parse_matmul_args(bindings, env, info)
        var dims = parse_dims(bindings, env, info)
        _matmul_tiled(ptrs[0], ptrs[1], ptrs[2], dims[0], dims[1], dims[2])
        return JsNumber.create(bindings, env, 0.0).value
    except:
        throw_js_error(env, "matmulTiled failed")
        return NapiValue()


# --- 4. Parallel: parallelize across tile rows --------------------------------
# Same tiled algorithm, but distributes row-tile strips across CPU cores.

comptime NUM_WORKERS = 4

def _matmul_parallel(
    a: UnsafePointer[Float64, MutAnyOrigin],
    b: UnsafePointer[Float64, MutAnyOrigin],
    c: UnsafePointer[Float64, MutAnyOrigin],
    M: Int, K: Int, N: Int,
):
    for i in range(M * N):
        c[i] = 0.0
    def worker(wid: Int) capturing:
        # Each worker handles tile-rows with stride = NUM_WORKERS
        var ii = wid * TILE_SIZE
        while ii < M:
            var i_end = ii + TILE_SIZE if ii + TILE_SIZE < M else M
            var pp = 0
            while pp < K:
                var p_end = pp + TILE_SIZE if pp + TILE_SIZE < K else K
                var jj = 0
                while jj < N:
                    var j_end = jj + TILE_SIZE if jj + TILE_SIZE < N else N
                    var tile_n = j_end - jj
                    for i in range(ii, i_end):
                        var row_c = i * N + jj
                        for p in range(pp, p_end):
                            var a_ip = a[i * K + p]
                            var row_b = p * N + jj
                            def compute[width: Int](j: Int) unified {read}:
                                var b_chunk = b.load[width=width](row_b + j)
                                var c_chunk = c.load[width=width](row_c + j)
                                c.store[width=width](row_c + j, c_chunk + a_ip * b_chunk)
                            vectorize[simd_width_of[DType.float64]()](tile_n, compute)
                    jj += TILE_SIZE
                pp += TILE_SIZE
            ii += NUM_WORKERS * TILE_SIZE
    parallelize[worker](NUM_WORKERS)


def matmul_parallel_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var bindings = CbArgs.get_bindings(env, info)
        var ptrs = parse_matmul_args(bindings, env, info)
        var dims = parse_dims(bindings, env, info)
        _matmul_parallel(ptrs[0], ptrs[1], ptrs[2], dims[0], dims[1], dims[2])
        return JsNumber.create(bindings, env, 0.0).value
    except:
        throw_js_error(env, "matmulParallel failed")
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

    var naive_ref = matmul_naive_fn
    var vec_ref = matmul_vectorized_fn
    var tiled_ref = matmul_tiled_fn
    var par_ref = matmul_parallel_fn

    try:
        var m = ModuleBuilder(env, exports, cb_data)
        m.method("matmulNaive", fn_ptr(naive_ref))
        m.method("matmulVectorized", fn_ptr(vec_ref))
        m.method("matmulTiled", fn_ptr(tiled_ref))
        m.method("matmulParallel", fn_ptr(par_ref))
        m.flush()
    except:
        pass

    return exports
