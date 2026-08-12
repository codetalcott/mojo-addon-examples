## packages/embed/src/embed.mojo — Python-interop embedding via MAX
##
## Calls packages/embed/embed.py (which loads sentence-transformers/all-MiniLM-L6-v2
## on MAX + H100) via Python interop, reading JS token buffers by raw address
## and writing embeddings back to the JS Float32Array.
##
## The Python engine is lazily constructed on first call and cached by
## embed.py as a module-level singleton — subsequent N-API calls reuse the
## compiled MAX graph (cold-start cost amortized over the session).

from std.memory.alloc import unsafe_alloc
from std.python import Python, PythonObject

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
from napi.bindings import Bindings
from napi.framework.js_int32 import JsInt32
from napi.framework.js_number import JsNumber
from napi.framework.js_typedarray import JsTypedArray
from napi.framework.js_ref import JsRef
from napi.framework.async_work import AsyncWork
from napi.framework.args import CbArgs
from napi.framework.register import fn_ptr, ModuleBuilder


comptime EMBED_DIM = 384


def _import_embed_module() raises -> PythonObject:
    # Ensure packages/embed/ is importable. We try both absolute (pod) and
    # relative paths so this works from the repo root on a laptop too.
    var sys = Python.import_module("sys")
    sys.path.insert(0, "/workspace/mojo-addon-examples/packages/embed")
    sys.path.insert(0, "packages/embed")
    return Python.import_module("embed")


def embed_tokens_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var b = CbArgs.get_bindings(env, info)
        # napi-mojo's CbArgs tops out at get_four; use get_argv for 5 args.
        var argv = unsafe_alloc[NapiValue](5)
        CbArgs.get_argv(b, env, info, 5, argv.as_unsafe_any_origin())

        # argv[0]: Int32Array of token IDs, shape [batch, seqLen]
        # argv[1]: Int32Array of attention mask, shape [batch, seqLen]
        # argv[2]: Int (batch)
        # argv[3]: Int (seqLen)
        # argv[4]: Float32Array dst, shape [batch, EMBED_DIM]
        var ids_ta = JsTypedArray(argv[unsafe_offset=0])
        var mask_ta = JsTypedArray(argv[unsafe_offset=1])
        var batch = Int(JsInt32.from_napi_value(b, env, argv[unsafe_offset=2]))
        var seq_len = Int(JsInt32.from_napi_value(b, env, argv[unsafe_offset=3]))
        var dst_ta = JsTypedArray(argv[unsafe_offset=4])
        argv.unsafe_free()

        # Dimension validation — cheap safety net, catches wrong sizes early.
        var ids_len = Int(ids_ta.length(b, env))
        var mask_len = Int(mask_ta.length(b, env))
        var dst_len = Int(dst_ta.length(b, env))
        var expected_tokens = batch * seq_len
        var expected_dst = batch * EMBED_DIM
        if ids_len < expected_tokens:
            raise Error("embedTokens: tokenIds too small")
        if mask_len < expected_tokens:
            raise Error("embedTokens: attentionMask too small")
        if dst_len < expected_dst:
            raise Error("embedTokens: dst too small")

        # Raw pointers to JS-owned buffers. We pass their integer addresses
        # to Python; Python uses ctypes.from_address to view them as numpy
        # arrays without copying.
        var ids_ptr = ids_ta.data_ptr(b, env)
        var mask_ptr = mask_ta.data_ptr(b, env)
        var dst_ptr = dst_ta.data_ptr(b, env)
        var ids_addr = Int(ids_ptr)
        var mask_addr = Int(mask_ptr)
        var dst_addr = Int(dst_ptr)

        var embed = _import_embed_module()
        _ = embed.embed_batch_from_addrs(
            ids_addr, mask_addr, dst_addr, batch, seq_len, EMBED_DIM
        )

        return JsNumber.create(b, env, 0.0).value
    except:
        # throw_js_error takes a StringLiteral only; the Python exception
        # details go to stderr. Inspect the capture file for what failed.
        throw_js_error(env, "embedTokens failed (see pod stderr)")
        return NapiValue(unsafe_from_address=Int(0))


# --- embedTokensAsync: non-blocking variant ---------------------------------
#
# Wraps the same embed_batch_from_addrs Python entry point in napi-mojo
# AsyncWork. Runs on the libuv threadpool so the Node event loop stays
# responsive under concurrent load.
#
# GIL handling: Mojo's Python interop acquires the GIL implicitly via
# CPython's thread-state machinery, so calls from a threadpool worker
# serialize naturally. The MAX InferenceSession inside embed.py is a
# module-level singleton — first call pays the 30s cold-start regardless
# of sync/async; subsequent calls reuse it.
#
# Pinning: the three TypedArrays (ids, mask, dst) are pinned via JsRef
# at submit time and unrefed in the complete callback. The raw addresses
# we pass into Python are stable as long as the ArrayBuffer isn't freed.


struct EmbedAsyncData(Movable):
    # JsRef is not Movable in napi-mojo 0.3.0, so we store raw NapiRef handles
    # (trivially copyable OpaquePointer) and wrap JsRef() at the delete site.
    @__allow_legacy_any_origin_fields
    var deferred: NapiDeferred
    @__allow_legacy_any_origin_fields
    var work: NapiAsyncWork
    @__allow_legacy_any_origin_fields
    var ids_ref: NapiRef
    @__allow_legacy_any_origin_fields
    var mask_ref: NapiRef
    @__allow_legacy_any_origin_fields
    var dst_ref: NapiRef
    var ids_addr: Int
    var mask_addr: Int
    var dst_addr: Int
    var batch: Int
    var seq_len: Int
    var had_error: Bool

    def __init__(
        out self,
        ids_ref: NapiRef,
        mask_ref: NapiRef,
        dst_ref: NapiRef,
        ids_addr: Int,
        mask_addr: Int,
        dst_addr: Int,
        batch: Int,
        seq_len: Int,
    ):
        self.deferred = NapiDeferred(unsafe_from_address=Int(0))
        self.work = NapiAsyncWork(unsafe_from_address=Int(0))
        self.ids_ref = ids_ref
        self.mask_ref = mask_ref
        self.dst_ref = dst_ref
        self.ids_addr = ids_addr
        self.mask_addr = mask_addr
        self.dst_addr = dst_addr
        self.batch = batch
        self.seq_len = seq_len
        self.had_error = False

    def __moveinit__(out self, deinit take: Self):
        self.deferred = take.deferred
        self.work = take.work
        self.ids_ref = take.ids_ref
        self.mask_ref = take.mask_ref
        self.dst_ref = take.dst_ref
        self.ids_addr = take.ids_addr
        self.mask_addr = take.mask_addr
        self.dst_addr = take.dst_addr
        self.batch = take.batch
        self.seq_len = take.seq_len
        self.had_error = take.had_error


def embed_async_execute(env: NapiEnv, data: OpaquePointer[MutAnyOrigin]):
    var ptr = data.unsafe_bitcast[EmbedAsyncData]()
    try:
        var embed = _import_embed_module()
        _ = embed.embed_batch_from_addrs(
            ptr[].ids_addr,
            ptr[].mask_addr,
            ptr[].dst_addr,
            ptr[].batch,
            ptr[].seq_len,
            EMBED_DIM,
        )
    except:
        ptr[].had_error = True


def embed_async_complete(
    env: NapiEnv, status: NapiStatus, data: OpaquePointer[MutAnyOrigin]
):
    var ptr = data.unsafe_bitcast[EmbedAsyncData]()
    try:
        JsRef(ptr[].ids_ref).delete(env)
    except:
        pass
    try:
        JsRef(ptr[].mask_ref).delete(env)
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
                env,
                ptr[].deferred,
                ptr[].work,
                "embedTokensAsync failed (see pod stderr)",
            )
    except:
        pass
    ptr.unsafe_deinit_pointee()
    ptr.unsafe_free()


def embed_tokens_async_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var b = CbArgs.get_bindings(env, info)
        var argv = unsafe_alloc[NapiValue](5)
        CbArgs.get_argv(b, env, info, 5, argv.as_unsafe_any_origin())

        var ids_ta = JsTypedArray(argv[unsafe_offset=0])
        var mask_ta = JsTypedArray(argv[unsafe_offset=1])
        var batch = Int(JsInt32.from_napi_value(b, env, argv[unsafe_offset=2]))
        var seq_len = Int(JsInt32.from_napi_value(b, env, argv[unsafe_offset=3]))
        var dst_ta = JsTypedArray(argv[unsafe_offset=4])

        var ids_len = Int(ids_ta.length(b, env))
        var mask_len = Int(mask_ta.length(b, env))
        var dst_len = Int(dst_ta.length(b, env))
        var expected_tokens = batch * seq_len
        var expected_dst = batch * EMBED_DIM
        if ids_len < expected_tokens:
            argv.unsafe_free()
            raise Error("embedTokensAsync: tokenIds too small")
        if mask_len < expected_tokens:
            argv.unsafe_free()
            raise Error("embedTokensAsync: attentionMask too small")
        if dst_len < expected_dst:
            argv.unsafe_free()
            raise Error("embedTokensAsync: dst too small")

        var ids_addr = Int(ids_ta.data_ptr(b, env))
        var mask_addr = Int(mask_ta.data_ptr(b, env))
        var dst_addr = Int(dst_ta.data_ptr(b, env))

        var ids_ref = JsRef.create(b, env, argv[unsafe_offset=0], UInt32(1)).handle
        var mask_ref = JsRef.create(b, env, argv[unsafe_offset=1], UInt32(1)).handle
        var dst_ref = JsRef.create(b, env, argv[unsafe_offset=4], UInt32(1)).handle
        argv.unsafe_free()

        var data_ptr = unsafe_alloc[EmbedAsyncData](1)
        data_ptr.unsafe_write(
            EmbedAsyncData(
                ids_ref,
                mask_ref,
                dst_ref,
                ids_addr,
                mask_addr,
                dst_addr,
                batch,
                seq_len,
            )
        )

        var exec_ref = embed_async_execute
        var comp_ref = embed_async_complete
        var aw = AsyncWork.queue(
            b,
            env,
            "embedTokensAsync",
            data_ptr.unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
            fn_ptr(exec_ref),
            fn_ptr(comp_ref),
        )
        data_ptr[].deferred = aw.deferred
        data_ptr[].work = aw.work
        return aw.value
    except:
        throw_js_error(env, "embedTokensAsync failed")
        return NapiValue(unsafe_from_address=Int(0))


def register_embed(mut m: ModuleBuilder, b: Bindings) raises:
    var et_ref = embed_tokens_fn
    var eta_ref = embed_tokens_async_fn
    m.method("embedTokens", fn_ptr(et_ref))
    m.method("embedTokensAsync", fn_ptr(eta_ref))
