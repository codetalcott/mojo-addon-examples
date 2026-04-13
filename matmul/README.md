# Matrix Multiply — Progressive Optimization

Four implementations showing Mojo's optimization story, from naive triple loop to SIMD + tiled + parallel. Each step maps to one Mojo feature with measurable speedup.

## Functions

| Export | Description | Mojo Feature |
|--------|-------------|-------------|
| `matmulNaive(a, b, out, M, K, N)` | Triple loop baseline | Compiler alone |
| `matmulVectorized(a, b, out, M, K, N)` | SIMD inner loop | `vectorize()` |
| `matmulTiled(a, b, out, M, K, N)` | 64x64 cache blocking | `vectorize()` + tiling |
| `matmulParallel(a, b, out, M, K, N)` | Multi-core tile rows | `parallelize()` |

All operate on row-major `Float64Array`s. JS passes pre-allocated output buffer for zero-allocation benchmarking.

## Results (M4 Mac, CPU)

*CPU-only — `vectorize()` + `parallelize()`, no GPU.*

| Step | 1024x1024 | 2048x2048 |
|------|-----------|-----------|
| JS baseline | 1x | 1x |
| Mojo naive | 2.2x | 1.9x |
| Mojo vectorized | 15.5x | 32.0x |
| Mojo tiled | 12.6x | 27.6x |
| **Mojo parallel** | **38.6x** | **91.4x** |

## Phase 3c.2 — Cached GPU matmul via `linalg.matmul` (NVIDIA H100 80GB HBM3)

Shipped as a separate [`matmul_cached.node`](addon_cached.mojo) addon that uses MAX's production [`linalg.matmul`](https://github.com/modular/modular/tree/main/max/kernels/src/linalg) kernel — tensor-core dispatch on NVIDIA (TF32 for FP32 inputs on H100) and Metal on Apple silicon. The kernel body is five lines:

```mojo
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul import matmul as linalg_matmul

var tt_a = TileTensor[dtype](a[].dev_data, row_major(Coord(Idx(M), Idx(K))))
var tt_b = TileTensor[dtype](b[].dev_data, row_major(Coord(Idx(K), Idx(N))))
var tt_c = TileTensor[dtype](dev_c,         row_major(Coord(Idx(M), Idx(N))))
linalg_matmul[target="gpu"](tt_c, tt_a, tt_b, Optional(ctx))
```

Upload A and B to the GPU once via `loadMatrixGpu`, then run `matmulHandle(hA, hB, dst)` many times. C is allocated per-call (output varies) but A and B stay resident.

```js
const hA = cached.loadMatrixGpu(aF32, M, K);
const hB = cached.loadMatrixGpu(bF32, K, N);
const dst = new Float32Array(M * N);
for (let i = 0; i < N_ITERS; i++) {
  cached.matmulHandle(hA, hB, dst);   // TF32 tensor-core matmul
}
cached.releaseMatrixGpu(hA);
cached.releaseMatrixGpu(hB);
```

### H100 results (FP32 inputs → TF32 tensor cores, validated 2026-04-12)

Three-path comparison on H100 80GB HBM3 via RunPod:

| Size  | JS       | Mojo CPU parallel | **GPU cached**       | `loadMatrixGpu` |
| ----- | -------- | ----------------- | -------------------- | --------------- |
| 256²  | 32.3 ms  | 1.00 ms (32×)     | **0.05 ms (606×)**   | 0.2 ms          |
| 512²  | 287 ms   | 13.0 ms (22×)     | **0.13 ms (2166×)**  | 0.8 ms          |
| 1024² | 5750 ms  | 62.0 ms (93×)     | **0.48 ms (12038×)** | 1.9 ms          |
| 2048² | 59591 ms | 561 ms (106×)     | **2.10 ms (28343×)** | 7.2 ms          |

**28343× JS at 2048²** is the largest speedup anywhere in this project — 25× bigger than Phase 3a's 1146× countByte result, and 267× faster than Mojo CPU parallel (AVX-512 on the H100 host's Xeon) on the same workload.

Break-even vs any cost of `loadMatrixGpu`: **1 call** at every size. Each subsequent call is pure kernel + D2H of the C matrix.

### Precision: TF32, not FP32-strict

On H100, FP32 inputs are converted to **TF32** (10-bit mantissa, same 8-bit exponent as FP32) inside the tensor cores, then accumulated and stored as FP32. This delivers ~494 TFLOPS at the cost of ~1e-3 relative error per multiply. For K summations the worst-case relative error compounds to ~K × 1e-3 — about 6% at K=64, potentially higher at K=2048.

The regression test ([test_cached.js](test_cached.js)) accommodates this with `rtol=1e-1`, `atol=1e-3`, and a 1% outlier budget (handles non-deterministic GPU scheduling). On M4 Metal the same test passes with much tighter actual error because Metal uses FP32 throughout.

**If you need FP32-strict results**, use the CPU `matmulParallel` path in `matmul.node`. The speed-vs-precision tradeoff is unavoidable at the tensor-core level — this is standard industry behavior, not a Mojo quirk.

### Why this is the biggest speedup in the project

1. **Matmul has high arithmetic intensity**: O(n³) compute on O(n²) data. At 2048² that's 17 GFLOPs of math on 16 MB of I/O — roughly 1000 flops per byte, perfectly suited for HBM + tensor cores.
2. **Persistent buffers eliminate the PCIe-bound weakness** seen in Phase 2 and partially in Phase 3b.1/3b.2. Per-call cost is kernel (~0.7 ms at 2048²) + D2H of the 16 MB C matrix (~1.4 ms at 12 GB/s).
3. **TF32 tensor cores are the dominant compute mode on H100**. 494 TFLOPS at TF32 vs ~60 TFLOPS at non-tensor-core FP32 — an 8× hardware advantage that materializes immediately when we call the right kernel.
4. **MAX's `linalg.matmul` is cuBLAS-equivalent** in performance. We're not benchmarking a hand-rolled kernel against CPU — we're benchmarking the production kernel library that MAX itself uses.

The previous "flagship" Phase 2 result was 91.4× at 2048² on M4 (CPU parallel). GPU cached on H100 is **310× faster than that**.

## Phase 3d — RAG-shape cached matmul + fused top-k

The 28,343× headline is 2048² square. The realistic Node.js market for this kernel is **local embedding retrieval** (query × corpus.T), where the shape is tall-skinny — `[B, d] × [d, N]` with B=1..256, d=768/1536, N=10k..1M. Lower arithmetic intensity than the square case, and the `[B, N]` D2H of scores adds per-call cost.

### searchHandle — fused matmul + per-row top-k

To make the RAG path a one-call primitive, `matmul_cached.node` exports `searchHandle`:

```js
const hQuery  = cached.loadMatrixGpu(queryEmbeddings, B, dim);   // [B, dim]
const hCorpus = cached.loadMatrixGpu(corpusT,         dim, N);   // [dim, N]
const idx     = new Uint32Array(B * k);
const scores  = new Float32Array(B * k);
cached.searchHandle(hQuery, hCorpus, idx, scores);   // k inferred from idx.length / B
```

Implementation: runs the existing `linalg.matmul` into a device scores buffer, D2Hs the `[B, N]` matrix, then a host-side min-heap top-k per row (O(N log k), descending by score). A GPU top-k kernel would save the full-scores D2H and is an obvious future optimization if profiling shows D2H dominates.

### M4 Metal results (local dev benchmark)

Run: `node matmul/matmul_rag.js` (append `--concurrency=100` for latency percentiles, `--full` for 1M-corpus shapes, `--no-ort` / `--no-hnsw` to skip baselines).

Three-path matmul comparison: single-threaded JS · `onnxruntime-node` CPU matmul (MLAS → Apple Accelerate) · our cached GPU path.

| Shape                              |   JS    | ORT CPU  | GPU cached | vs ORT |
| ---------------------------------- | ------: | -------: | ---------: | -----: |
| `[1, 768] × [768, 100k]` (RAG)     | 90.6 ms | 19.6 ms  | **5.6 ms** | **3.5× win** |
| `[1, 1536] × [1536, 100k]`         |  (skip) | 38.2 ms  | **6.9 ms** | **5.6× win** |
| `[64, 768] × [768, 100k]`          |  (skip) | **30.3 ms** | 147.8 ms | 4.9× ORT win |
| `[256, 768] × [768, 100k]` offline |  (skip) | **58.9 ms** | 590.5 ms | 10× ORT win  |

Honest reading:

- **Single-query RAG (B=1)**: cached GPU is 3–5× faster than the best Node CPU option. 5–7 ms latency is below what a hosted vector-DB round-trip costs. This is the latency play.
- **Batched matmul on M4 Metal**: ORT CPU wins decisively. MLAS/Accelerate sustains ~700 GFLOP/s on batched `MatMul`; our `linalg.matmul[target="gpu"]` path stays flat at ~67 GFLOP/s regardless of batch size on M4, and the `[B, N]` D2H scales with B. The throughput play requires a GPU with tensor cores and HBM — H100 numbers pending.
- **Event-loop stability**: `p99 / p50 = 2.1×` at single-query under a 100-call burst (`--concurrency=100`) — no starvation under sustained sync dispatch.

### RAG product comparison — exact GPU vs approximate HNSW

When the question is "top-k semantic search" rather than "matmul backend," the right competitor is `hnswlib-node` (approximate nearest neighbor). Bench builds an HNSW index once (4–5 min at N=100k × d=768, amortized like `loadMatrixGpu`), then sweeps the query-time `ef` parameter (higher = better recall, slower).

`[1, 768] × [768, 100k]` at k=10, recall measured against exact ground truth:

| Baseline            | Latency  | Recall@10 | Notes |
| ------------------- | -------: | --------: | ----- |
| HNSW `ef=100`       |  1.1 ms  |     0.10  | HNSW's default; fast but nearly useless on our synthetic data |
| HNSW `ef=500`       |  5.3 ms  |     0.10  | Roughly equal latency to GPU exact, 10× worse recall |
| HNSW `ef=2000`      | 18.2 ms  |     0.70  | Still sub-perfect recall; GPU exact is 3× faster |
| **GPU `searchHandle`** | **6.0 ms** | **1.00** | Exact; fused matmul + host min-heap top-k |

At larger batches (B=64, 256), GPU exact pulls further ahead: at k=10 recall 0.65–0.70, HNSW ef=2000 runs **7× slower** than GPU exact.

**Important caveat on the HNSW numbers:** the bench uses L2-normalized random vectors, which are worst-case for graph-based ANN — concentration of measure in high dim makes "top-10 nearest" a near-tie among thousands of candidates, and HNSW's graph can't discriminate. On real sentence-transformer embeddings (clustered, lower effective dimensionality), HNSW at `ef=100` routinely hits recall > 0.95. The takeaway: the exact path's value is recall, not raw latency; users who can tolerate approximate results should still benchmark HNSW on their actual data.

The ORT baseline uses a dynamic-shape MatMul graph at [matmul/fixtures/matmul_dyn.onnx](fixtures/matmul_dyn.onnx) (118 bytes, regenerable via the one-liner in the file header).

### M4 Metal memory ceiling

`node matmul/matmul_rag.js --full` attempts `[1, 768] × [768, 1_000_000]` (3 GB corpus). On M4 Metal this crashes inside `loadMatrixGpu` without unwinding to JS — Metal can't cleanly allocate a 3 GB device buffer from the unified memory pool while `onnxruntime-node` already holds a copy. **100k is the practical corpus ceiling for d=768 on M4 Metal.** H100 with 80 GB HBM3 wouldn't have the problem; until there's a hardware target with headroom, treat `--full` as H100-only.

### End-to-end RAG demo

See [`examples/rag-demo/search.js`](../examples/rag-demo/search.js) — ~80 lines of Node wrapping `searchHandle` into a `GpuIndex.search(query, k)` method. Generates a synthetic 10k × 768 corpus, runs a self-similarity query, prints the top 10 at ~7 ms/query on M4 Metal. Swap `makeSyntheticCorpus` for a loader over your real precomputed embeddings.

## Build & Run

```bash
# Original CPU matmul (four progressive optimizations):
pixi run bash matmul/build.sh
node matmul/matmul.js

# Cached GPU matmul (Phase 3c.2) + RAG primitive (Phase 3d):
pixi run bash matmul/build_cached.sh
node matmul/matmul_cached.js
node matmul/matmul_rag.js --concurrency=100
node examples/rag-demo/search.js
```
