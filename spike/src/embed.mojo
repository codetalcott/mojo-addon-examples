## spike/src/embed.mojo — MAX graph + embedTokens N-API binding (DAY 1 STUB)
##
## Day 1: embedTokens fills dst with zeros. Proves the N-API arg unpacking
## works end-to-end and the build toolchain compiles against napi-mojo.
##
## Day 2: replace _stub_forward() with a real MAX graph forward pass over
## MiniLM-L6-v2. Pattern to study: node_modules/napi-mojo/src/addon/async_ops.mojo
## for async wrapping (will want this after Day 5 passes).
##
## DAY 1 GOAL: `node spike/test-roundtrip.js` prints a (batch × 384) array of
## zeros with no thrown errors. Nothing more.

from std.memory import memset

from napi.types import NapiEnv, NapiValue
from napi.error import throw_js_error
from napi.bindings import Bindings
from napi.framework.js_int32 import JsInt32
from napi.framework.js_number import JsNumber
from napi.framework.js_typedarray import JsTypedArray
from napi.framework.args import CbArgs
from napi.framework.register import fn_ptr, ModuleBuilder


alias EMBED_DIM = 384  # MiniLM-L6-v2


def _stub_forward(
    dst_bytes: UnsafePointer[Byte, MutAnyOrigin],
    batch: Int,
):
    # Day 1 placeholder: zero-fill. Day 2 replaces with MAX graph forward.
    var n_bytes = batch * EMBED_DIM * 4
    memset(dst_bytes, 0, n_bytes)


def embed_tokens_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var b = CbArgs.get_bindings(env, info)
        var args = CbArgs.get_five(b, env, info)

        # args[0]: Int32Array of token IDs, shape [batch, seqLen]
        # args[1]: Int32Array of attention mask, shape [batch, seqLen]
        # args[2]: Int (batch)
        # args[3]: Int (seqLen)
        # args[4]: Float32Array dst, shape [batch, EMBED_DIM]
        var ids_ta = JsTypedArray(args[0])
        var mask_ta = JsTypedArray(args[1])
        var batch = Int(JsInt32.from_napi_value(b, env, args[2]))
        var seq_len = Int(JsInt32.from_napi_value(b, env, args[3]))
        var dst_ta = JsTypedArray(args[4])

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

        var dst_ptr = dst_ta.data_ptr(b, env)
        _stub_forward(dst_ptr, batch)
        return JsNumber.create(b, env, 0.0).value
    except:
        throw_js_error(env, "embedTokens failed")
        return NapiValue()


def register_embed(mut m: ModuleBuilder, b: Bindings) raises:
    var et_ref = embed_tokens_fn
    m.method("embedTokens", fn_ptr(et_ref))
