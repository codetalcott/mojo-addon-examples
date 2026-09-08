"""Embedding backends for semnotes. DRAFT — see README.md.

Two backends, chosen by SEMNOTES_BACKEND:

- ``hash`` (the default): a deterministic feature-hashing embedder — words
  and character 3-grams hashed into 384 signed dimensions, L2-normalized.
  Pure stdlib, microseconds per text, and stable across processes, which is
  what lets smoke.py assert on rankings with no model on the machine. It is
  a *lexical* embedding: real semantic similarity needs the ``max``
  backend, but every wire, storage and ranking property of the app is
  identical under either.
- ``max``: MiniLM-L6-v2 through this repo's own engine —
  ``packages/embed/embed.py``'s ``EmbeddingEngine``, a MAX graph compiled
  with ``InferenceSession`` (SEMNOTES_DEVICE=cpu|gpu, default cpu).
  Tokenization is HF ``tokenizers`` loaded from the engine's own model
  snapshot, so the two stay in vocabulary lockstep. Needs the environment
  that runs packages/embed (max, numpy, transformers, tokenizers,
  huggingface_hub). 384 dims — the same dimensionality as the hash
  backend on purpose, so the two are drop-in-swappable against one
  database schema (do not mix backends in one database file: the spaces
  are unrelated, so cross-backend scores are noise).

SEMNOTES_SPIN_MS busy-spins that many milliseconds inside every hash-backend
embed call. It exists for smoke.py's isolation phase: the property under
test is the server's (a saturated handler pool must not stall the loop's
health lane), and the knob gives the hash backend a model's worth of
GIL-holding latency without a model's dependencies. It has no effect on the
``max`` backend, whose latency is real.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import sys
import time
from pathlib import Path

DIM = 384
MAX_TOKENS = 256


def _spin_ms() -> float:
    try:
        return float(os.environ.get("SEMNOTES_SPIN_MS", "0"))
    except ValueError:
        return 0.0


class HashEmbedder:
    """Deterministic feature-hashing embedding. Lexical, not semantic."""

    name = "hash"
    dim = DIM

    def embed(self, text: str) -> list[float]:
        deadline = None
        spin = _spin_ms()
        if spin > 0:
            deadline = time.perf_counter() + spin / 1000.0
        vec = [0.0] * DIM
        words = "".join(c.lower() if c.isalnum() else " " for c in text).split()
        feats = list(words)
        for w in words:
            padded = f"#{w}#"
            feats.extend(padded[i : i + 3] for i in range(len(padded) - 2))
        for f in feats:
            h = hashlib.blake2b(f.encode("utf-8"), digest_size=8).digest()
            (n,) = struct.unpack("<Q", h)
            idx = n % DIM
            sign = 1.0 if (n >> 63) else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        if deadline is not None:
            while time.perf_counter() < deadline:
                pass  # busy: emulate a GIL-holding model, not a releasing sleep
        return vec


class MaxEmbedder:
    """MiniLM-L6-v2 via packages/embed's MAX EmbeddingEngine."""

    name = "max"
    dim = DIM

    def __init__(self) -> None:
        embed_dir = os.environ.get(
            "SEMNOTES_EMBED_DIR",
            str(Path(__file__).resolve().parents[2] / "packages" / "embed"),
        )
        if embed_dir not in sys.path:
            sys.path.insert(0, embed_dir)
        import embed  # noqa: PLC0415 — packages/embed, resolved above
        from tokenizers import Tokenizer  # noqa: PLC0415

        device = os.environ.get("SEMNOTES_DEVICE", "cpu")
        self._engine = embed.get_engine(device)
        self._tokenizer = Tokenizer.from_file(
            str(self._engine.model_dir / "tokenizer.json")
        )
        self._tokenizer.enable_truncation(max_length=MAX_TOKENS)
        import numpy as np  # noqa: PLC0415 — a dependency of embed already

        self._np = np

    def embed(self, text: str) -> list[float]:
        np = self._np
        enc = self._tokenizer.encode(text)
        ids = np.asarray([enc.ids], dtype=np.int64)
        mask = np.asarray([enc.attention_mask], dtype=np.int64)
        out = self._engine.embed_batch_l2(ids, mask)[0]
        if out.shape[0] != DIM:
            raise RuntimeError(f"engine returned {out.shape[0]} dims, expected {DIM}")
        return [float(x) for x in out]


def get_embedder():
    backend = os.environ.get("SEMNOTES_BACKEND", "hash").strip().lower()
    if backend == "hash":
        return HashEmbedder()
    if backend == "max":
        return MaxEmbedder()
    raise RuntimeError(f"unknown SEMNOTES_BACKEND {backend!r} (hash|max)")
