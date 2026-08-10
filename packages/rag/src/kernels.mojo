## packages/rag/src/kernels.mojo — GPU matmul + top-k retrieval primitives
##
## Public API (registered onto ModuleBuilder by register_gpu_linalg):
##   loadMatrixGpu(f32Array, rows, cols)         -> External handle (CachedMatrix)
##   matmulHandle(hA, hB, dstF32Array)           -> fills dst in place (C = A × B)
##   searchHandle(hA, hB, outIdxU32, outScoresF32)
##       fused matmul + per-row top-k (RAG query × corpus.T workloads).
##       k inferred from outIdx.length / A.rows. Results sorted descending.
##   releaseMatrixGpu(h)                         -> tombstone; memory freed on GC
##
## Uses MAX's linalg.matmul which dispatches to tensor cores on NVIDIA
## (TF32 for FP32 inputs on H100, FP16 tensor cores for FP16) and Metal on
## Apple silicon. Persistent device buffers live inside CachedMatrix handles
## and are freed by the JsExternal.create_typed[T] finalizer when the JS
## handle is GC'd (typed helper pattern landed in 0.3.0).
##
## GpuState (DeviceContext) is stored as per-env instance_data; on hosts
## without a GPU, set_instance_data raises and call sites throw a clear
## error. Registration always succeeds so module load never fails.

from std.math import ceildiv
from std.memory import unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from layout import Coord, TileTensor, row_major
from linalg.matmul import matmul as linalg_matmul

from napi.types import (
    NapiEnv,
    NapiValue,
    NapiStatus,
    NapiDeferred,
    NapiAsyncWork,
    NapiRef,
    NAPI_OK,
)
from napi.error import throw_js_error
from napi.bindings import NapiBindings, Bindings
from napi.framework.instance_data import set_instance_data, get_instance_data
from napi.framework.js_number import JsNumber
from napi.framework.js_int32 import JsInt32
from napi.framework.js_typedarray import JsTypedArray
from napi.framework.js_external import JsExternal
from napi.framework.js_ref import JsRef
from napi.framework.async_work import AsyncWork
from napi.framework.args import CbArgs
from napi.framework.register import fn_ptr, ModuleBuilder


comptime dtype = DType.float32

# NOTE: a hand-rolled tall-skinny matmul kernel was attempted (shared-memory
# A cache + coalesced B reads across threadgroup-consecutive j) on the
# theory that linalg.matmul[target="gpu"]'s 67 GFLOP/s ceiling on M4 Metal
# was due to kernel selection targeting square-ish shapes. The custom kernel
# landed 1.7× slower at batch-64/256 and ~10% slower at single-query — MAX's
# kernel selector already dispatches well on Metal; the 67 GFLOP/s is a
# hardware ceiling (memory bandwidth + no tensor cores on Apple Silicon for
# FP32), not a software one. H100 HBM3 + TF32 tensor cores is where the
# throughput story flips.


# --- GpuState ----------------------------------------------------------------

struct GpuState(Movable):
    var ctx: DeviceContext

    def __init__(out self, var ctx: DeviceContext):
        self.ctx = ctx^


def _get_gpu_state(
    b: Bindings, env: NapiEnv
) raises -> Pointer[GpuState, MutAnyOrigin]:
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
    src_bytes: Pointer[Byte, MutAnyOrigin],
    rows: Int,
    cols: Int,
) raises -> CachedMatrix:
    var num_elems = rows * cols
    var num_bytes = num_elems * 4

    var dev_data = ctx.enqueue_create_buffer[dtype](num_elems)
    var staging = ctx.enqueue_create_host_buffer[dtype](num_elems)
    var staging_ptr = staging.unsafe_ptr().unsafe_bitcast[Byte]()
    unsafe_memcpy(dest=staging_ptr, src=src_bytes, count=num_bytes)
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
        return NapiValue(unsafe_from_address=Int(0))


# --- matmulHandle: C = A × B using linalg.matmul ----------------------------

def _matmul_cached(
    ctx: DeviceContext,
    a: Pointer[CachedMatrix, MutAnyOrigin],
    b: Pointer[CachedMatrix, MutAnyOrigin],
    dst_bytes: Pointer[Byte, MutAnyOrigin],
) raises:
    var M = a[].rows
    var K = a[].cols
    var N = b[].cols
    var c_elems = M * N

    # Per-call C buffer (device + pinned host for D2H). linalg.matmul writes
    # every element of C (no accumulation from initial contents), so the
    # enqueue_fill(0.0) that used to live here was wasted bandwidth on
    # tall-skinny RAG shapes — at [256, 768] × [768, 100k] the 102 MB zero
    # fill was ~15% of per-call time on M4 Metal.
    var dev_c = ctx.enqueue_create_buffer[dtype](c_elems)
    var host_c = ctx.enqueue_create_host_buffer[dtype](c_elems)

    # Wrap persistent A, B and per-call C as TileTensors.
    var tt_a = TileTensor[dtype](a[].dev_data, row_major(Coord(M, K)))
    var tt_b = TileTensor[dtype](b[].dev_data, row_major(Coord(K, N)))
    var tt_c = TileTensor[dtype](dev_c, row_major(Coord(M, N)))

    linalg_matmul[target="gpu"](tt_c, tt_a, tt_b, Optional(ctx))

    ctx.enqueue_copy(host_c, dev_c)
    ctx.synchronize()

    var host_ptr = host_c.unsafe_ptr().unsafe_bitcast[Byte]()
    unsafe_memcpy(dest=dst_bytes, src=host_ptr, count=c_elems * 4)


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

        if a[].cols != b_mat[].rows:
            raise Error("matmulHandle: dimension mismatch (A.cols != B.rows)")

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
        return NapiValue(unsafe_from_address=Int(0))


# --- searchHandle: fused matmul + per-row top-k -----------------------------
#
# RAG primitive. Runs C = A × B on the GPU, then on the host picks the k
# largest scores per row of C along with their column indices. Returns
# results sorted descending.
#
# Why host-side top-k (not a GPU kernel): for typical RAG shapes
# (B = 1..256, N = 10k..1M, k = 10..100), the D2H of C [B, N] still has to
# happen — the matmul output is on device. A GPU top-k would save that D2H
# and is worth revisiting if bench numbers show it dominates; for now the
# simpler correct thing is a min-heap on the host, O(N log k) per row.

def _topk_row(
    row_scores: Pointer[Float32, MutAnyOrigin],
    n: Int,
    k: Int,
    out_scores: Pointer[Float32, MutAnyOrigin],
    out_idx: Pointer[UInt32, MutAnyOrigin],
):
    # Build a min-heap of size k seeded with the first k scores, then replace
    # the root whenever a later element beats it. At the end heap-sort into
    # descending order so callers get rank-ordered results.
    if k <= 0:
        return
    var heap_n = k if k < n else n

    for i in range(heap_n):
        out_scores[unsafe_offset=i] = row_scores[unsafe_offset=i]
        out_idx[unsafe_offset=i] = UInt32(i)

    var i = heap_n // 2 - 1
    while i >= 0:
        _sift_down(out_scores, out_idx, heap_n, i)
        i -= 1

    for j in range(heap_n, n):
        var v = row_scores[unsafe_offset=j]
        if v > out_scores[unsafe_offset=0]:
            out_scores[unsafe_offset=0] = v
            out_idx[unsafe_offset=0] = UInt32(j)
            _sift_down(out_scores, out_idx, heap_n, 0)

    # Heap-sort with a min-heap: repeatedly swap root (current min) to the
    # shrinking end, sift down. Each iteration places the smallest remaining
    # element at the high end, so the final array is sorted descending.
    var end = heap_n - 1
    while end > 0:
        var tmps = out_scores[unsafe_offset=0]
        var tmpi = out_idx[unsafe_offset=0]
        out_scores[unsafe_offset=0] = out_scores[unsafe_offset=end]
        out_idx[unsafe_offset=0] = out_idx[unsafe_offset=end]
        out_scores[unsafe_offset=end] = tmps
        out_idx[unsafe_offset=end] = tmpi
        _sift_down(out_scores, out_idx, end, 0)
        end -= 1

    # If k > n, zero-fill the tail so callers don't read uninitialized memory.
    for j in range(heap_n, k):
        out_scores[unsafe_offset=j] = 0.0
        out_idx[unsafe_offset=j] = 0


def _sift_down(
    scores: Pointer[Float32, MutAnyOrigin],
    idx: Pointer[UInt32, MutAnyOrigin],
    n: Int,
    start: Int,
):
    var root = start
    while True:
        var child = 2 * root + 1
        if child >= n:
            return
        if child + 1 < n and scores[unsafe_offset=child + 1] < scores[unsafe_offset=child]:
            child += 1
        if scores[unsafe_offset=root] <= scores[unsafe_offset=child]:
            return
        var tmps = scores[unsafe_offset=root]
        var tmpi = idx[unsafe_offset=root]
        scores[unsafe_offset=root] = scores[unsafe_offset=child]
        idx[unsafe_offset=root] = idx[unsafe_offset=child]
        scores[unsafe_offset=child] = tmps
        idx[unsafe_offset=child] = tmpi
        root = child


def _search_cached(
    ctx: DeviceContext,
    a: Pointer[CachedMatrix, MutAnyOrigin],
    b: Pointer[CachedMatrix, MutAnyOrigin],
    k: Int,
    out_idx: Pointer[UInt32, MutAnyOrigin],
    out_scores: Pointer[Float32, MutAnyOrigin],
) raises:
    var M = a[].rows
    var K = a[].cols
    var N = b[].cols
    var c_elems = M * N

    var dev_c = ctx.enqueue_create_buffer[dtype](c_elems)
    var host_c = ctx.enqueue_create_host_buffer[dtype](c_elems)

    var tt_a = TileTensor[dtype](a[].dev_data, row_major(Coord(M, K)))
    var tt_b = TileTensor[dtype](b[].dev_data, row_major(Coord(K, N)))
    var tt_c = TileTensor[dtype](dev_c, row_major(Coord(M, N)))

    linalg_matmul[target="gpu"](tt_c, tt_a, tt_b, Optional(ctx))

    ctx.enqueue_copy(host_c, dev_c)
    ctx.synchronize()

    var host_ptr = host_c.unsafe_ptr().as_unsafe_any_origin()
    for row in range(M):
        _topk_row(
            host_ptr.unsafe_offset(row * N),
            N,
            k,
            out_scores.unsafe_offset(row * k),
            out_idx.unsafe_offset(row * k),
        )


def search_handle_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var b = CbArgs.get_bindings(env, info)
        var args = CbArgs.get_four(b, env, info)

        var a = JsExternal.get_typed[CachedMatrix](
            b, env, args[0], "searchHandle A"
        )
        if a[].released:
            raise Error("searchHandle: handle A has been released")

        var b_mat = JsExternal.get_typed[CachedMatrix](
            b, env, args[1], "searchHandle B"
        )
        if b_mat[].released:
            raise Error("searchHandle: handle B has been released")

        if a[].cols != b_mat[].rows:
            raise Error("searchHandle: dimension mismatch (A.cols != B.rows)")

        var idx_ta = JsTypedArray(args[2])
        var scores_ta = JsTypedArray(args[3])
        var idx_len = Int(idx_ta.length(b, env))
        var scores_len = Int(scores_ta.length(b, env))
        if idx_len != scores_len:
            raise Error("searchHandle: indices and scores must have same length")

        var M = a[].rows
        if idx_len % M != 0:
            raise Error("searchHandle: output length must be a multiple of A.rows")
        var k = idx_len // M
        if k <= 0:
            raise Error("searchHandle: k must be > 0")

        var idx_ptr = idx_ta.data_ptr(b, env).unsafe_bitcast[UInt32]()
        var scores_ptr = scores_ta.data_ptr(b, env).unsafe_bitcast[Float32]()

        var state = _get_gpu_state(b, env)
        _search_cached(state[].ctx, a, b_mat, k, idx_ptr, scores_ptr)
        return JsNumber.create(b, env, 0.0).value
    except:
        throw_js_error(env, "searchHandle failed")
        return NapiValue(unsafe_from_address=Int(0))


# --- Async variants of matmulHandle / searchHandle --------------------------
#
# These wrap the existing _matmul_cached / _search_cached helpers in napi-mojo
# AsyncWork so the GPU work runs on the libuv threadpool instead of blocking
# the Node event loop. Gate G2: event-loop p99 jitter < 5 ms under 100-
# concurrent load.
#
# Cross-thread safety:
#   - Externals (CachedMatrix handles) and the dst TypedArray are pinned via
#     JsRef.create(..., 1) at submit time and unreffed in the complete
#     callback. This prevents JS GC from freeing the backing store while
#     the execute callback is mid-flight on the threadpool.
#   - GpuState lives as per-env instance_data and is stable for the env
#     lifetime; we pass a pointer, not a copy.
#   - CUDA stream ops on a single DeviceContext serialize on the CUDA side
#     when called from multiple threads; concurrent async submits enqueue
#     work sequentially on the stream but do not trample. The event-loop
#     benefit comes from unblocking the JS thread, not from true GPU
#     parallelism — that would require per-stream contexts.


struct MatmulAsyncData(Movable):
    # JsRef is not Movable in napi-mojo 0.3.0, so we store the raw NapiRef
    # handles (trivially copyable OpaquePointer) and reconstitute JsRef() at
    # the complete-callback delete site. NapiRef is an alias for
    # OpaquePointer[MutAnyOrigin].
    @__allow_legacy_any_origin_fields
    var deferred: NapiDeferred
    @__allow_legacy_any_origin_fields
    var work: NapiAsyncWork
    @__allow_legacy_any_origin_fields
    var a_ref: NapiRef
    @__allow_legacy_any_origin_fields
    var b_ref: NapiRef
    @__allow_legacy_any_origin_fields
    var dst_ref: NapiRef
    @__allow_legacy_any_origin_fields
    var a_ptr: Pointer[CachedMatrix, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var b_ptr: Pointer[CachedMatrix, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var dst_ptr: Pointer[Byte, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var state_ptr: Pointer[GpuState, MutAnyOrigin]
    var had_error: Bool

    def __init__(
        out self,
        a_ref: NapiRef,
        b_ref: NapiRef,
        dst_ref: NapiRef,
        a_ptr: Pointer[CachedMatrix, MutAnyOrigin],
        b_ptr: Pointer[CachedMatrix, MutAnyOrigin],
        dst_ptr: Pointer[Byte, MutAnyOrigin],
        state_ptr: Pointer[GpuState, MutAnyOrigin],
    ):
        self.deferred = NapiDeferred(unsafe_from_address=Int(0))
        self.work = NapiAsyncWork(unsafe_from_address=Int(0))
        self.a_ref = a_ref
        self.b_ref = b_ref
        self.dst_ref = dst_ref
        self.a_ptr = a_ptr
        self.b_ptr = b_ptr
        self.dst_ptr = dst_ptr
        self.state_ptr = state_ptr
        self.had_error = False

    def __moveinit__(out self, deinit take: Self):
        self.deferred = take.deferred
        self.work = take.work
        self.a_ref = take.a_ref
        self.b_ref = take.b_ref
        self.dst_ref = take.dst_ref
        self.a_ptr = take.a_ptr
        self.b_ptr = take.b_ptr
        self.dst_ptr = take.dst_ptr
        self.state_ptr = take.state_ptr
        self.had_error = take.had_error


def matmul_async_execute(env: NapiEnv, data: OpaquePointer[MutAnyOrigin]):
    var ptr = data.unsafe_bitcast[MatmulAsyncData]()
    try:
        _matmul_cached(
            ptr[].state_ptr[].ctx,
            ptr[].a_ptr,
            ptr[].b_ptr,
            ptr[].dst_ptr,
        )
    except:
        ptr[].had_error = True


def matmul_async_complete(
    env: NapiEnv, status: NapiStatus, data: OpaquePointer[MutAnyOrigin]
):
    var ptr = data.unsafe_bitcast[MatmulAsyncData]()
    try:
        JsRef(ptr[].a_ref).delete(env)
    except:
        pass
    try:
        JsRef(ptr[].b_ref).delete(env)
    except:
        pass
    try:
        JsRef(ptr[].dst_ref).delete(env)
    except:
        pass
    try:
        if status == NAPI_OK and not ptr[].had_error:
            var result_val = JsNumber.create(env, 0.0)
            AsyncWork.resolve(
                env, ptr[].deferred, ptr[].work, result_val.value
            )
        else:
            AsyncWork.reject_with_error(
                env, ptr[].deferred, ptr[].work, "matmulHandleAsync failed"
            )
    except:
        pass
    ptr.unsafe_deinit_pointee()
    ptr.unsafe_free()


def matmul_handle_async_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var b = CbArgs.get_bindings(env, info)
        var args = CbArgs.get_three(b, env, info)

        var a_ptr = JsExternal.get_typed[CachedMatrix](
            b, env, args[0], "matmulHandleAsync A"
        )
        if a_ptr[].released:
            raise Error("matmulHandleAsync: handle A has been released")

        var b_ptr = JsExternal.get_typed[CachedMatrix](
            b, env, args[1], "matmulHandleAsync B"
        )
        if b_ptr[].released:
            raise Error("matmulHandleAsync: handle B has been released")

        if a_ptr[].cols != b_ptr[].rows:
            raise Error(
                "matmulHandleAsync: dimension mismatch (A.cols != B.rows)"
            )

        var dst_ta = JsTypedArray(args[2])
        var dst_len = Int(dst_ta.length(b, env))
        var expected_len = a_ptr[].rows * b_ptr[].cols
        if dst_len < expected_len:
            raise Error("matmulHandleAsync: dst buffer too small")
        var dst_raw = dst_ta.data_ptr(b, env)

        var state = _get_gpu_state(b, env)

        var a_ref = JsRef.create(b, env, args[0], UInt32(1)).handle
        var b_ref = JsRef.create(b, env, args[1], UInt32(1)).handle
        var dst_ref = JsRef.create(b, env, args[2], UInt32(1)).handle

        var data_ptr = unsafe_alloc[MatmulAsyncData](1)
        data_ptr.unsafe_write(
            MatmulAsyncData(
                a_ref, b_ref, dst_ref, a_ptr, b_ptr, dst_raw, state
            )
        )

        var exec_ref = matmul_async_execute
        var comp_ref = matmul_async_complete
        var aw = AsyncWork.queue(
            b,
            env,
            "matmulHandleAsync",
            data_ptr.unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
            fn_ptr(exec_ref),
            fn_ptr(comp_ref),
        )
        data_ptr[].deferred = aw.deferred
        data_ptr[].work = aw.work
        return aw.value
    except:
        throw_js_error(env, "matmulHandleAsync failed")
        return NapiValue(unsafe_from_address=Int(0))


struct SearchAsyncData(Movable):
    # See MatmulAsyncData for the NapiRef-instead-of-JsRef rationale.
    @__allow_legacy_any_origin_fields
    var deferred: NapiDeferred
    @__allow_legacy_any_origin_fields
    var work: NapiAsyncWork
    @__allow_legacy_any_origin_fields
    var a_ref: NapiRef
    @__allow_legacy_any_origin_fields
    var b_ref: NapiRef
    @__allow_legacy_any_origin_fields
    var idx_ref: NapiRef
    @__allow_legacy_any_origin_fields
    var scores_ref: NapiRef
    @__allow_legacy_any_origin_fields
    var a_ptr: Pointer[CachedMatrix, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var b_ptr: Pointer[CachedMatrix, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var idx_ptr: Pointer[UInt32, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var scores_ptr: Pointer[Float32, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var state_ptr: Pointer[GpuState, MutAnyOrigin]
    var k: Int
    var had_error: Bool

    def __init__(
        out self,
        a_ref: NapiRef,
        b_ref: NapiRef,
        idx_ref: NapiRef,
        scores_ref: NapiRef,
        a_ptr: Pointer[CachedMatrix, MutAnyOrigin],
        b_ptr: Pointer[CachedMatrix, MutAnyOrigin],
        idx_ptr: Pointer[UInt32, MutAnyOrigin],
        scores_ptr: Pointer[Float32, MutAnyOrigin],
        state_ptr: Pointer[GpuState, MutAnyOrigin],
        k: Int,
    ):
        self.deferred = NapiDeferred(unsafe_from_address=Int(0))
        self.work = NapiAsyncWork(unsafe_from_address=Int(0))
        self.a_ref = a_ref
        self.b_ref = b_ref
        self.idx_ref = idx_ref
        self.scores_ref = scores_ref
        self.a_ptr = a_ptr
        self.b_ptr = b_ptr
        self.idx_ptr = idx_ptr
        self.scores_ptr = scores_ptr
        self.state_ptr = state_ptr
        self.k = k
        self.had_error = False

    def __moveinit__(out self, deinit take: Self):
        self.deferred = take.deferred
        self.work = take.work
        self.a_ref = take.a_ref
        self.b_ref = take.b_ref
        self.idx_ref = take.idx_ref
        self.scores_ref = take.scores_ref
        self.a_ptr = take.a_ptr
        self.b_ptr = take.b_ptr
        self.idx_ptr = take.idx_ptr
        self.scores_ptr = take.scores_ptr
        self.state_ptr = take.state_ptr
        self.k = take.k
        self.had_error = take.had_error


def search_async_execute(env: NapiEnv, data: OpaquePointer[MutAnyOrigin]):
    var ptr = data.unsafe_bitcast[SearchAsyncData]()
    try:
        _search_cached(
            ptr[].state_ptr[].ctx,
            ptr[].a_ptr,
            ptr[].b_ptr,
            ptr[].k,
            ptr[].idx_ptr,
            ptr[].scores_ptr,
        )
    except:
        ptr[].had_error = True


def search_async_complete(
    env: NapiEnv, status: NapiStatus, data: OpaquePointer[MutAnyOrigin]
):
    var ptr = data.unsafe_bitcast[SearchAsyncData]()
    try:
        JsRef(ptr[].a_ref).delete(env)
    except:
        pass
    try:
        JsRef(ptr[].b_ref).delete(env)
    except:
        pass
    try:
        JsRef(ptr[].idx_ref).delete(env)
    except:
        pass
    try:
        JsRef(ptr[].scores_ref).delete(env)
    except:
        pass
    try:
        if status == NAPI_OK and not ptr[].had_error:
            var result_val = JsNumber.create(env, 0.0)
            AsyncWork.resolve(
                env, ptr[].deferred, ptr[].work, result_val.value
            )
        else:
            AsyncWork.reject_with_error(
                env, ptr[].deferred, ptr[].work, "searchHandleAsync failed"
            )
    except:
        pass
    ptr.unsafe_deinit_pointee()
    ptr.unsafe_free()


def search_handle_async_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var b = CbArgs.get_bindings(env, info)
        var args = CbArgs.get_four(b, env, info)

        var a_ptr = JsExternal.get_typed[CachedMatrix](
            b, env, args[0], "searchHandleAsync A"
        )
        if a_ptr[].released:
            raise Error("searchHandleAsync: handle A has been released")

        var b_ptr = JsExternal.get_typed[CachedMatrix](
            b, env, args[1], "searchHandleAsync B"
        )
        if b_ptr[].released:
            raise Error("searchHandleAsync: handle B has been released")

        if a_ptr[].cols != b_ptr[].rows:
            raise Error(
                "searchHandleAsync: dimension mismatch (A.cols != B.rows)"
            )

        var idx_ta = JsTypedArray(args[2])
        var scores_ta = JsTypedArray(args[3])
        var idx_len = Int(idx_ta.length(b, env))
        var scores_len = Int(scores_ta.length(b, env))
        if idx_len != scores_len:
            raise Error(
                "searchHandleAsync: indices and scores must have same length"
            )

        var M = a_ptr[].rows
        if idx_len % M != 0:
            raise Error(
                "searchHandleAsync: output length must be a multiple of"
                " A.rows"
            )
        var k = idx_len // M
        if k <= 0:
            raise Error("searchHandleAsync: k must be > 0")

        var idx_raw = idx_ta.data_ptr(b, env).unsafe_bitcast[UInt32]()
        var scores_raw = scores_ta.data_ptr(b, env).unsafe_bitcast[Float32]()

        var state = _get_gpu_state(b, env)

        var a_ref = JsRef.create(b, env, args[0], UInt32(1)).handle
        var b_ref = JsRef.create(b, env, args[1], UInt32(1)).handle
        var idx_ref = JsRef.create(b, env, args[2], UInt32(1)).handle
        var scores_ref = JsRef.create(b, env, args[3], UInt32(1)).handle

        var data_ptr = unsafe_alloc[SearchAsyncData](1)
        data_ptr.unsafe_write(
            SearchAsyncData(
                a_ref,
                b_ref,
                idx_ref,
                scores_ref,
                a_ptr,
                b_ptr,
                idx_raw,
                scores_raw,
                state,
                k,
            )
        )

        var exec_ref = search_async_execute
        var comp_ref = search_async_complete
        var aw = AsyncWork.queue(
            b,
            env,
            "searchHandleAsync",
            data_ptr.unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
            fn_ptr(exec_ref),
            fn_ptr(comp_ref),
        )
        data_ptr[].deferred = aw.deferred
        data_ptr[].work = aw.work
        return aw.value
    except:
        throw_js_error(env, "searchHandleAsync failed")
        return NapiValue(unsafe_from_address=Int(0))


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
        return NapiValue(unsafe_from_address=Int(0))


# --- Registration -----------------------------------------------------------

def register_gpu_linalg(mut m: ModuleBuilder, b: Bindings) raises:
    # Try to initialize a GPU context for this env. If no accelerator is
    # available we swallow the error here — module load still succeeds and
    # method callers see a clear "requires a GPU" error at first call time.
    try:
        var ctx = DeviceContext()
        set_instance_data(b, m.env, GpuState(ctx^))
    except:
        pass

    var lmg_ref = load_matrix_gpu_fn
    var mh_ref = matmul_handle_fn
    var sh_ref = search_handle_fn
    var mha_ref = matmul_handle_async_fn
    var sha_ref = search_handle_async_fn
    var rmg_ref = release_matrix_gpu_fn

    m.method("loadMatrixGpu", fn_ptr(lmg_ref))
    m.method("matmulHandle", fn_ptr(mh_ref))
    m.method("searchHandle", fn_ptr(sh_ref))
    m.method("matmulHandleAsync", fn_ptr(mha_ref))
    m.method("searchHandleAsync", fn_ptr(sha_ref))
    m.method("releaseMatrixGpu", fn_ptr(rmg_ref))
