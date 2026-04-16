## spike/src/embed.mojo — Python-interop embedding via MAX (Day 3)
##
## Calls spike/embed.py (which loads sentence-transformers/all-MiniLM-L6-v2
## on MAX + H100) via Python interop, reading JS token buffers by raw address
## and writing embeddings back to the JS Float32Array.
##
## The Python engine is lazily constructed on first call and cached by
## embed.py as a module-level singleton — subsequent N-API calls reuse the
## compiled MAX graph (cold-start cost amortized over the session).

from std.memory import alloc
from std.python import Python, PythonObject

from napi.types import NapiEnv, NapiValue
from napi.error import throw_js_error
from napi.bindings import Bindings
from napi.framework.js_int32 import JsInt32
from napi.framework.js_number import JsNumber
from napi.framework.js_typedarray import JsTypedArray
from napi.framework.args import CbArgs
from napi.framework.register import fn_ptr, ModuleBuilder


alias EMBED_DIM = 384


def _import_embed_module() raises -> PythonObject:
    # Ensure spike/ is importable. We try both absolute (pod) and relative
    # paths so this works from the repo root on a laptop too.
    var sys = Python.import_module("sys")
    sys.path.insert(0, "/workspace/mojo-addon-examples/spike")
    sys.path.insert(0, "spike")
    return Python.import_module("embed")


def embed_tokens_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var b = CbArgs.get_bindings(env, info)
        # napi-mojo's CbArgs tops out at get_four; use get_argv for 5 args.
        var argv = alloc[NapiValue](5)
        CbArgs.get_argv(b, env, info, 5, argv)

        # argv[0]: Int32Array of token IDs, shape [batch, seqLen]
        # argv[1]: Int32Array of attention mask, shape [batch, seqLen]
        # argv[2]: Int (batch)
        # argv[3]: Int (seqLen)
        # argv[4]: Float32Array dst, shape [batch, EMBED_DIM]
        var ids_ta = JsTypedArray(argv[0])
        var mask_ta = JsTypedArray(argv[1])
        var batch = Int(JsInt32.from_napi_value(b, env, argv[2]))
        var seq_len = Int(JsInt32.from_napi_value(b, env, argv[3]))
        var dst_ta = JsTypedArray(argv[4])
        argv.free()

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
        return NapiValue()


def register_embed(mut m: ModuleBuilder, b: Bindings) raises:
    var et_ref = embed_tokens_fn
    m.method("embedTokens", fn_ptr(et_ref))
