"""packages/embed/embed.py — Python-side embedding engine.

Loads sentence-transformers/all-MiniLM-L6-v2 via MAX (using vendored pipeline
code from packages/embed/bert_graph.py to sidestep max.pipelines.lib dep hell)
and exposes a simple `embed_batch(ids_np, mask_np) -> np.ndarray` for the Mojo
addon to call via Python interop.

Usage (Python side, for testing):
    import embed  # from packages/embed
    engine = embed.EmbeddingEngine(device='gpu')       # or 'cpu'
    embeddings = engine.embed_batch(token_ids, mask)   # returns (B, 384) fp32

Usage (Mojo via Python interop):
    var sys = Python.import_module("sys")
    sys.path.append("/workspace/mojo-addon-examples/packages/embed")
    var embed_mod = Python.import_module("embed")
    var engine = embed_mod.EmbeddingEngine("gpu")
    var result = engine.embed_batch(ids_np, mask_np)
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file as safetensors_load
from transformers import AutoConfig

from max import driver
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef

# Vendored from max.pipelines.architectures.bert
from bert_graph import BertModelConfig, build_graph
from bert_weight_adapter import convert_safetensor_state_dict

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384


def _pick_device(kind: str = "gpu") -> driver.Device:
    if kind == "gpu":
        try:
            return driver.Accelerator()
        except Exception as e:
            log.warning(f"Accelerator failed ({e}); falling back to CPU")
    return driver.CPU()


def _load_weights(model_dir: Path) -> dict:
    """Load safetensors weights from a downloaded HF snapshot."""
    # MiniLM ships model.safetensors; some variants use pytorch_model.bin
    st_path = model_dir / "model.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(
            f"No model.safetensors in {model_dir}. MiniLM should ship this; "
            f"check that snapshot_download completed."
        )
    # safetensors.numpy returns a dict of {name: np.ndarray} directly.
    return safetensors_load(str(st_path))


class _NumpyWeights:
    """Minimal shim making a numpy dict look like max.graph.weights.Weights
    for the purposes of convert_safetensor_state_dict.

    Upstream expects each value to have `.data() -> WeightData`. For our spike
    we just return the numpy array directly — MAX's InferenceSession.load
    accepts numpy arrays in weights_registry (they get DLPack-converted)."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def data(self):
        return self._arr


class EmbeddingEngine:
    """One-time-loaded, reusable MiniLM-L6-v2 embedder.

    Lifetime: the caller keeps this instance alive for the process. Each
    embed_batch() runs a GPU forward pass using the compiled model.
    """

    def __init__(self, device: str = "gpu", cache_dir: str | None = None):
        t0 = time.perf_counter()
        self.device = _pick_device(device)
        log.info(f"device: {self.device}")

        # Resolve cache dir — use HF_HOME if set (pod's persistent volume)
        if cache_dir is None:
            cache_dir = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")

        log.info(f"fetching {MODEL_ID} (cache: {cache_dir})")
        self.model_dir = Path(snapshot_download(MODEL_ID, cache_dir=cache_dir))

        # Raw weights (numpy arrays)
        raw_weights = _load_weights(self.model_dir)
        log.info(f"loaded {len(raw_weights)} weight tensors from safetensors")

        # Apply BERT name adapter (HF -> MAX naming convention)
        shimmed = {k: _NumpyWeights(v) for k, v in raw_weights.items()}
        self.state_dict = convert_safetensor_state_dict(shimmed)
        log.info(f"after adapter: {len(self.state_dict)} tensors")

        # HuggingFace config (model architecture params)
        hf_config = AutoConfig.from_pretrained(self.model_dir)

        # Build MAX graph
        device_ref = DeviceRef.GPU() if self.device.label == "gpu" else DeviceRef.CPU()
        config = BertModelConfig(
            huggingface_config=hf_config,
            dtype=DType.float32,
            device=device_ref,
            pool_embeddings=True,   # mean-pooled + attention-mask-normalized
        )
        log.info(f"building graph (vocab={hf_config.vocab_size}, hidden={hf_config.hidden_size})")
        graph = build_graph(config, self.state_dict)

        # Compile
        log.info("compiling with InferenceSession.load...")
        t1 = time.perf_counter()
        self.session = InferenceSession(devices=[self.device])
        self.model = self.session.load(graph, weights_registry=self.state_dict)
        t2 = time.perf_counter()

        log.info(f"ready  (compile: {t2-t1:.1f}s, total init: {t2-t0:.1f}s)")

    def embed_batch_l2(self, token_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Embed + L2-normalize. Matches sentence-transformers default."""
        out = self.embed_batch(token_ids, attention_mask)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)

    def embed_batch(self, token_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Embed a batch of token-id sequences.

        Args:
            token_ids: int32 or int64 array of shape [B, seq_len]
            attention_mask: int32 or int64 array of shape [B, seq_len]
                (1 for real tokens, 0 for padding)

        Returns:
            float32 array of shape [B, EMBED_DIM]. Already mean-pooled
            (attention-mask-weighted). Caller does L2-normalize if desired.
        """
        # BERT graph expects int64 ids + float32 mask
        ids = token_ids.astype(np.int64, copy=False)
        mask = attention_mask.astype(np.float32, copy=False)

        ids_buf = driver.Buffer.from_numpy(ids).to(self.device)
        mask_buf = driver.Buffer.from_numpy(mask).to(self.device)

        result = self.model.execute(ids_buf, mask_buf)
        out = result[0]  # Buffer on self.device

        # For now we return numpy (D2H bounce). Zero-copy via DLPack is
        # Phase 2 once Mojo-side wrapping is wired up.
        return out.to_numpy()


# --- Singleton + Mojo-facing entrypoint ------------------------------------

_ENGINE_SINGLETON: EmbeddingEngine | None = None


def get_engine(device: str = "gpu") -> EmbeddingEngine:
    """Lazy singleton. First call pays the model download + compile cost;
    subsequent calls return the cached instance. Mojo reuses this across
    N-API invocations for the duration of the Node process."""
    global _ENGINE_SINGLETON
    if _ENGINE_SINGLETON is None:
        _ENGINE_SINGLETON = EmbeddingEngine(device=device)
    return _ENGINE_SINGLETON


def embed_batch_from_addrs(
    ids_addr: int,
    mask_addr: int,
    dst_addr: int,
    batch: int,
    seq_len: int,
    embed_dim: int,
) -> int:
    """Called from Mojo via Python interop. Reads int32 token IDs + int32
    attention mask from raw JS-owned addresses, runs the MAX embedder,
    L2-normalizes, writes float32 output into the JS-owned dst buffer.

    Returns 0 on success; raises on failure (which surfaces as a Python
    exception that Mojo's try/except converts to a JS error)."""
    n = batch * seq_len

    # View the JS Int32Arrays as numpy without copying. We upcast to int64
    # for BERT and float32 for the mask in a second pass (makes a copy, but
    # at n=128*64=8k elements that's ~30μs — negligible).
    ids_raw = np.ctypeslib.as_array(
        (ctypes.c_int32 * n).from_address(ids_addr)
    ).reshape(batch, seq_len)
    mask_raw = np.ctypeslib.as_array(
        (ctypes.c_int32 * n).from_address(mask_addr)
    ).reshape(batch, seq_len)

    ids = ids_raw.astype(np.int64, copy=True)
    mask = mask_raw.astype(np.float32, copy=True)

    engine = get_engine()
    result = engine.embed_batch_l2(ids, mask)  # (batch, embed_dim) fp32

    if result.shape != (batch, embed_dim):
        raise RuntimeError(
            f"unexpected embedding shape: got {result.shape}, expected ({batch}, {embed_dim})"
        )
    if not result.flags["C_CONTIGUOUS"]:
        result = np.ascontiguousarray(result)

    # Copy fp32 embeddings into the JS-owned Float32Array.
    ctypes.memmove(
        dst_addr,
        result.ctypes.data_as(ctypes.c_void_p),
        batch * embed_dim * 4,
    )
    return 0


def _demo():
    """Minimal CLI: `python packages/embed/embed.py` runs a sanity check."""
    engine = EmbeddingEngine(device=os.environ.get("SPIKE_DEVICE", "gpu"))
    # Dummy tokenized input — two short fake sequences
    # (real tokens come from @huggingface/transformers JS tokenizer in Day 4)
    ids = np.array([
        [101, 7592, 2088, 102, 0, 0, 0, 0],
        [101, 1996, 4248, 2829, 4419, 102, 0, 0],
    ], dtype=np.int64)
    mask = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
    ], dtype=np.int64)
    t0 = time.perf_counter()
    embs = engine.embed_batch(ids, mask)
    t1 = time.perf_counter()
    print(f"embeddings shape: {embs.shape}  dtype: {embs.dtype}")
    print(f"norms: {np.linalg.norm(embs, axis=1)}")
    print(f"latency: {(t1-t0)*1000:.1f}ms for batch-{embs.shape[0]}")
    print(f"first 8 of row 0: {embs[0, :8]}")


if __name__ == "__main__":
    _demo()
