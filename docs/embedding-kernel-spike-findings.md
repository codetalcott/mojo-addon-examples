# Embedding Kernel Spike — Findings

Spike question: can a Mojo N-API addon run MiniLM-L6-v2 embeddings on H100 fast enough to compose with [`packages/retrieve`](../packages/retrieve/) inside one Node process? Answer: yes — 1.44 ms per-query (batch-1, seq-32, 1k corpus, warm-path), ~2× faster than an ONNX + hnswlib CPU reference. Productized as [`packages/embed/`](../packages/embed/).

## Expectation

Four gates defined success. Any failure → NO-GO.

| Gate | Target | Rationale |
|---|---|---|
| **F1 model loadability** | MiniLM-L6-v2 loads and runs on H100 from Mojo | No point benchmarking a model we can't load |
| **F2 latency** | Batch-1 ≤ 5 ms p50; batch-64 ≤ 20 ms p50 | Must beat ONNX CPU embedding for a single query |
| **F3 tokenize** | ≤ 20 ms on common inputs | JS tokenize can't dominate the end-to-end budget |
| **F4 correctness** | Cosine ≥ 0.995 vs sentence-transformers reference | Numerical parity within fp32 tolerance |

Planned 10 working days.

## Process

### Infrastructure: Lambda → RunPod pivot

Lambda Cloud had no H100 capacity at spike start (2026-04-16). Switched to RunPod Secure Cloud H100 SXM at ~$2.99/hr. Kept [`scripts/lambda-bench.sh`](../scripts/lambda-bench.sh) in the tree for when capacity returns.

RunPod pods use a persistent Network Volume that caches the pixi env + model weights. Cold-start on a fresh pod is ~30 s with cached volume, ~5 min without. Per-launch cost ~$0.15.

### Architectural pivot: no Mojo-native MAX

The original plan assumed a Mojo-native model loader (`from max.graph import Graph, ops`). **None exists in MAX v26:**

- `max.graph.Graph`, `max.nn.Module`, `max.engine.InferenceSession` — Python-only
- `max/c/model.h` C API exists but expects pre-compiled graphs from Python tooling
- Compiled `.mojopkg` files expose primitive ops + scheduling, not model loaders

The sanctioned path to MAX from Mojo is **Python interop** via `from std.python import Python`. The spike pivoted to this path on Day 1.

### End-to-end architecture

```text
JS query
  → @huggingface/transformers tokenize
  → N-API embedTokens(ids, mask, dst)
  → Mojo embed_tokens_fn
  → Python.import_module("embed")
  → EmbeddingEngine.embed_batch_l2  (MAX graph on H100)
  → fp32 embeddings
  → ctypes.memmove into JS Float32Array

(composes with:)
  packages/retrieve loadMatrixGpu → matmulHandle → searchHandle (exact cosine + top-k)
```

Both `embed.node` and `retrieve.node` load into the **same Node process** with **separate CUDA contexts** — the kernel-factory composition.

## Findings

### MAX v26 API cheat sheet

The API differs from what pretrained model training data suggests. Use these:

| What you want | v26 syntax |
|---|---|
| GPU device handle | `driver.Accelerator()` (NOT `driver.GPU()`) |
| Graph device spec | `DeviceRef.GPU()` in `TensorType` |
| Host→GPU transfer | `driver.Buffer.from_numpy(x).to(dev)` |
| Compile | `session = InferenceSession(devices=[dev]); model = session.load(graph)` |
| Execute | `model.execute(buf1, buf2, ...)` — inputs must be on target device |
| GPU output | List of `driver.Buffer`; `__dlpack__()` + `__dlpack_device__()` work |
| Load weights | `session.load(graph, weights_registry={"layer.weight": tensor, ...})` |
| Python-in-Mojo import | `var max_engine = Python.import_module("max.engine")` |

**Apple Silicon caveat:** `driver.Accelerator()` on M4 raises `"Not implemented for device: Apple M4"`. MAX's CPU path works locally, but all GPU iteration happens on RunPod.

### Vendored BERT graph (avoided ~300 MB of deps)

MAX ships a BERT pipeline supporting `sentence-transformers/all-MiniLM-L6-v2` — no need to hand-build the transformer graph. But `max.pipelines` carries heavy deps (pydantic, pillow, msgspec, and a growing list for diffusion/vision/LLM models).

For an embedding-only spike, we vendored the two files actually needed — [`packages/embed/bert_graph.py`](../packages/embed/bert_graph.py) + [`packages/embed/bert_weight_adapter.py`](../packages/embed/bert_weight_adapter.py) — with a minimal duck-typed `BertModelConfig`. Insulates from upstream registry churn.

Weight loading: `safetensors.numpy.load_file` (not `safetensors.torch`, which requires PyTorch). `transformers`'s "PyTorch was not found" warning is cosmetic — we only use `AutoConfig.from_pretrained()` for BERT architecture params.

L2-normalization is not automatic in MAX's `BertModel`; added explicitly in [`packages/embed/embed.py`](../packages/embed/embed.py) for sentence-transformers parity.

### Mojo↔Python zero-copy mechanics

- Mojo gets raw pointers from N-API via `JsTypedArray.data_ptr`
- `Int(ptr)` implicitly converts the pointer to integer address
- Python reads via `numpy.ctypeslib.as_array((c_int32 * n).from_address(addr))` — zero-copy view of JS memory
- Output: `ctypes.memmove(dst_addr, result.ctypes.data_as(c_void_p), nbytes)` writes into the JS Float32Array directly
- `EmbeddingEngine` is a module-level singleton; first N-API call pays the 29.6 s cold-start, subsequent calls reuse the compiled graph

**DLPack zero-copy through GPU pointers is a deferred optimization.** `__dlpack__()` / `__dlpack_device__()` round-trip cleanly end-to-end; wrapping as `DeviceBuffer(ctx, ptr, count, owning=False)` from Mojo would eliminate the H2D/D2H bounce. Current bounce is ~40 μs at d=384 — negligible vs ~1.6 ms embed + 0.09 ms search.

### Benchmarks — H100 warm-path, 50-iter p50

```text
CORPUS 1000 docs
  batch-1  seq-32    total=1.44 ms  (p95=2.13, p99=2.30)
  batch-1  seq-128   total=2.68 ms  (p95=3.17, p99=3.34)
  batch-64 seq-32    total=8.64 ms
  batch-64 seq-128   total=17.88 ms

CORPUS 10000 docs
  batch-1  seq-32    total=1.81 ms
  batch-1  seq-128   total=2.89 ms
  batch-64 seq-128   total=18.67 ms

Corpus-embed throughput: 18,448 docs/sec (batch-64, warm)
Cold-start (one-time): 29.6 s  (CUDA kernel JIT)
```

Per-stage breakdown (batch-1, seq-32, 1k corpus):

| Stage | p50 |
|---|---|
| Tokenize (JS, `@huggingface/transformers`) | 0.06 ms |
| Embed (MAX GPU forward + D2H) | 1.30 ms |
| Search (`packages/retrieve` matmul + top-k) | 0.084 ms |
| **Total** | **1.44 ms** |

Raw capture: [`docs/runpod-day5-bench-20260417T001138Z.txt`](runpod-day5-bench-20260417T001138Z.txt).

### Gate verdicts

| Gate | Target | Measured | Result |
|---|---|---|---|
| F1 model loadability | H100 forward runs | Python-interop path validated; DLPack round-trip confirmed | **PASS** |
| F2 latency batch-1 | ≤ 5 ms | 1.44 ms seq-32, 2.68 ms seq-128 | **PASS** (3× headroom) |
| F2 latency batch-64 | ≤ 20 ms | 8.64 ms seq-32, 17.88 ms seq-128 | **PASS** |
| F3 tokenize | ≤ 20 ms | 0.06 ms | **PASS** (300× headroom) |
| F4 correctness | cosine ≥ 0.995 | min 0.999990 across 100 sanity sentences | **PASS** (fp32-limit) |

### vs. CPU reference

Reference: `@huggingface/transformers` ONNX CPU + `hnswlib-node` ef=100 on the same sanity set.

| | Spike (MAX H100) | Reference (ONNX CPU + hnswlib) |
|---|---|---|
| Per-query total (batch-1 seq-32, 1k corpus) | **1.44 ms** | 3.13 ms |
| Corpus embed throughput (warm) | **18,448 docs/sec** | 143 docs/sec |
| Recall | 1.0 (exact) | 1.0 (clustered corpus) |
| Cold-start | 29.6 s | 1.1 s |

**~2× per query, ~130× for corpus indexing** (warm). Cold-start is the honest tradeoff — acceptable for long-running services, fatal for scripts/CLI.

Search stays ahead as corpus grows despite being O(N) exact matmul: 0.084 ms on 1k → 0.158 ms on 10k (H100 memory bandwidth dominates). At 100k expect ~1–2 ms per query total.

Numerical parity: fp32 BERT forward matches ONNX Runtime's fp32 BERT forward to 6 decimal places. Worst offender (sentence 36, "Comments explain why, not what.") at cosine 0.999990 — ULP-level accumulation-order differences, not model bugs.

## Budget & verdict

- H100 SXM pod-hours: ~3 at $2.99/hr
- Total spend: **~$10**
- Days elapsed: 2 (planned: 10)

**Verdict: GO.** Productized as [`packages/embed/`](../packages/embed/) — promoted `spike/` to `packages/embed/`, wired async N-API variants (`embedTokensAsync` / `matmulHandleAsync` / `searchHandleAsync`), ran MS-MARCO warm-path benchmarks.
