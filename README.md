# Mojo Addon Examples

High-performance Node.js addon examples built with [napi-mojo](https://github.com/codetalcott/napi-mojo). Each example demonstrates Mojo's SIMD `vectorize()` and `parallelize()` through the N-API bridge, with benchmarks against pure JavaScript.

## What's in this repo

- [`examples/`](examples/) — Standalone Mojo + Node.js kernels (matmul, SIMD search, stats, image, wyhash) with per-example benchmarks on M4 Metal and H100.
- [`packages/rag/`](packages/rag/) — `@qkstat/rag`, GPU exact-retrieval primitives (matmul + per-row top-k). 0.06 ms top-10 at recall 1.0 on MS-MARCO 10k (H100). Pre-release.
- [`packages/embed/`](packages/embed/) — `@qkstat/embed`, MiniLM-L6-v2 embeddings on H100 via MAX + Python interop. Composes with `packages/rag` in one Node.js process. 1.36 ms p50 embed+search on 1k corpus. Pre-release.
- [`spikes/mojo-runtime/`](spikes/mojo-runtime/) + [`docs/mojo-runtime-isolation-spike-findings.md`](docs/mojo-runtime-isolation-spike-findings.md) — Tiered-imports experiment isolating which Mojo runtime libraries a binary links against (five tiers, `ldd` captures).
- [`scripts/`](scripts/) + [`docs/cloud-benchmark-runbook.md`](docs/cloud-benchmark-runbook.md) — RunPod orchestration for reproducing H100 benchmarks (~$1, ~30 min per run).

## Examples

> **Benchmarking scope:** The primary CPU numbers in each table below are measured on an Apple M4. GPU rows report the M4's integrated GPU via Metal 4. A separate [H100 Cloud Benchmark Results](#h100-cloud-benchmark-results) section below reports results from an NVIDIA H100 80GB HBM3 (rented via RunPod). **Headline finding from the cloud runs**: for the single-shot GPU APIs that upload on every call, Mojo CPU SIMD beats Mojo GPU on every benchmark — on both M4 Metal *and* H100 CUDA — because the workload is PCIe-bound rather than HBM-bound. **Phase 3a flipped this** with a new persistent device buffer API (`loadGpu` / `countByteHandle` / `releaseGpu` in [simd-search](#simd-text-search--byte-level-pattern-matching)): on H100 the cached path reaches **1030× JS on 105 MB countByte, beating CPU SIMD by 30×**. See the [H100 Cloud Benchmark Results](#h100-cloud-benchmark-results) section and per-addon READMEs for the full teardown.

### Matrix Multiply — Progressive Optimization

Four implementations showing Mojo's optimization story, from naive triple loop to SIMD + tiled + parallel:

```
node examples/matmul/matmul.js
```

**Results (M4 Mac, CPU, Float64):**

| Step | 1024x1024 | 2048x2048 | Mojo Feature |
|------|-----------|-----------|-------------|
| JS baseline | 1x | 1x | — |
| Mojo naive | 2.2x | 1.9x | Same algorithm, better compiler |
| Mojo vectorized | 15.5x | 32.0x | `vectorize()` — SIMD inner loop |
| Mojo tiled | 12.6x | 27.6x | Cache-friendly 64x64 blocking |
| **Mojo parallel** | **38.6x** | **91.4x** | `parallelize()` — multi-core |

### SIMD Text Search — Byte-Level Pattern Matching

SIMD byte scanning that's impossible to express in pure JavaScript. Three functions: `countByte`, `countLines`, `searchAll` (single and multi-byte patterns).

```
node examples/simd-search/search.js
```

**Results (M4 Mac, Buffer/Uint8Array):**

| Function | 1MB | 16MB | 100MB | Mojo Feature |
|----------|-----|------|-------|-------------|
| **countByte CPU** | **19.2x** | **52.4x** | **67.6x** | SIMD XOR + reduce, `parallelize()` |
| countByte GPU Metal | 0.9x | 1.0x | 2.3x | Tree reduction, shared memory partials |
| countLines | 18.5x | 50.1x | 65.3x | Same kernel, byte=0x0A |
| searchAll (1-byte) | 2.5x | 2.8x | 3.1x | Two-pass: SIMD count + collect |
| searchAll (multi-byte) | 2.0x | 2.3x | 2.5x | First+last byte SIMD filter |

`countByteGpu()` is published honestly: at 100MB it's 2.3× JS vs CPU SIMD's 67.6× — a 30× gap. Byte-scan kernels on integrated GPUs are copy-bound rather than compute-bound; the M4 CPU's direct DRAM access wins decisively. Output matches the CPU path exactly. See [examples/simd-search/README.md](examples/simd-search/README.md) for why, and expected-to-flip-on-H100 notes.

### Statistics — SIMD Aggregation

Compute `{mean, stddev, min, max, p50, p95, p99}` on Float64Arrays in a single call. SIMD reductions + parallel accumulation + quickselect percentiles.

```
node examples/stats/stats.js
```

**Results (M4 Mac, Float64):**

| Function | 100K | 1M | 10M | Mojo Feature |
|----------|------|-----|-----|-------------|
| **stats() CPU** | **4.2x** | **5.8x** | **6.7x** | SIMD reduce_add/min/max, `parallelize()` |
| stats() GPU Metal | 2.1x | 4.0x | 4.2x | Shared-memory tree reduction via `DeviceContext` (Float32 internal) |
| histogram() CPU | 3.7x | 3.9x | 4.0x | SIMD min/max range detection |

`statsGpu()` runs on the M4's integrated GPU via Mojo's Metal 4 backend. Kernels are Float32 (Metal constraint); the H2D cast and final Float64 reduction happen on the host. See [examples/stats/README.md](examples/stats/README.md) for precision notes.

### Image Processing — Pixel Operations

Four RGBA pixel operations on Uint8Arrays: `grayscale`, `brightness`, `threshold`, `blur`. Integer-approximation grayscale, fixed-point brightness, separable box blur with parallel horizontal + vertical passes.

```
node examples/image/image.js
```

**Results (M4 Mac, RGBA Uint8Array):**

| Function | 720p | 1080p | 4K | Mojo Feature |
|----------|------|-------|-----|-------------|
| grayscale CPU | 6.8x | 5.4x | **6.8x** | Integer `(77R+150G+29B)>>8`, `parallelize()` |
| grayscale GPU Metal | 0.6x | 0.6x | 0.8x | One-pixel-per-thread elementwise kernel |
| brightness | 4.7x | 5.1x | 5.0x | Fixed-point multiply + clamp, `parallelize()` |
| threshold | 5.5x | 5.2x | **6.5x** | Grayscale + compare, `parallelize()` |
| **blur(r=5)** | 6.8x | **10.1x** | 5.6x | Separable box blur, parallel rows + cols |

`grayscaleGpu()` is published despite being **slower than JS** because it illustrates an important limit: on integrated GPUs (M4's unified memory architecture), the H2D/D2H "copies" are pure overhead with no bandwidth benefit. For a trivial per-pixel kernel the copy cost dominates. Output matches the CPU path byte-for-byte. On a discrete GPU with real HBM the ordering is expected to flip — see [examples/image/README.md](examples/image/README.md).

### wyhash — Fast Non-Cryptographic Hash

Match C hash performance in ~50 lines of Mojo. `wyHash` returns BigInt (full 64-bit), `wyHash64` returns Number (lossy but no BigInt allocation overhead). The speed comes from 128-bit folded multiplies via Mojo's native `DType.uint128`.

```
node examples/wyhash/hash.js
```

**Results (M4 Mac, CPU, Buffer):**

| Function | 1KB | 64KB | 1MB | 16MB | Mojo Feature |
|----------|-----|------|-----|------|-------------|
| **wyHash** (BigInt) | 3.7x | **52.9x** | **65.9x** | **66.2x** | 128-bit folded multiply |
| wyHash64 (Number) | 2.9x | 45.5x | 57.8x | 58.7x | Same kernel, Number return |

## H100 Cloud Benchmark Results

We ran the three GPU-enabled addons (stats, image, simd-search) on an **NVIDIA H100 80GB HBM3** via RunPod (~$1.99/hr) to see whether the M4 Metal story — GPU losing to CPU SIMD — would flip on Tier-1 discrete GPU hardware. **It did not.** On every benchmark, Mojo CPU SIMD running on the H100 host's Xeon (AVX-512) beat Mojo GPU running on the H100.

The reason matters more than the numbers. These addons call the GPU as one-shot N-API functions against **host-resident** data. Every call does `alloc → H2D copy → kernel → D2H copy → free`. On a PCIe Gen4 x16 link (~12 GB/s effective), PCIe is the bottleneck for data transfer, and the H100's 3 TB/s HBM3 bandwidth is effectively wasted because the data never lives on-device long enough to amortize it. Meanwhile the Xeon host's AVX-512 reads the same data at ~30 GB/s directly from DRAM. For low-arithmetic-intensity kernels (reductions, elementwise, byte scanning) on a single-call API, CPU wins.

This is the opposite of the narrative in most "port to GPU" benchmarks, and it's the most useful finding from the cloud run.

### Stats on H100 (Float64 arrays, ratios vs V8 on the same host)

| Size | H100 Mojo SIMD | H100 Mojo GPU |
|------|----------------|---------------|
| 100K | 5.0× | 4.7× |
| 1M | 5.3× | 5.2× |
| 10M | **8.3×** | 7.6× |

Two factors drag stats on GPU: (1) the Float64→Float32 cast runs as a scalar host loop at [examples/stats/addon.mojo:327](examples/stats/addon.mojo#L327) — at 10M elements this alone is tens of ms per call — and (2) stats does two separate passes (`{sum,min,max}` then `sum_sq_diff(mean)`), each with its own allocation + copy cycle, doubling per-call overhead.

### Image grayscale on H100 (4K RGBA)

| Size | H100 Mojo SIMD | H100 Mojo GPU |
|------|----------------|---------------|
| 720p | 3.2× | 2.6× |
| 1080p | 4.0× | 2.1× |
| 4K | **4.3×** | 1.4× |

The GPU gets *worse* relative to CPU as size grows. At 720p both paths bottleneck on per-call overhead so they're close; at 4K, CPU SIMD scales with DRAM bandwidth (~30 GB/s effective via AVX-512) while the GPU is PCIe-capped on 33 MB H2D + 33 MB D2H each call. Grayscale is the canonical memory-bandwidth-bound elementwise kernel where PCIe decides everything.

### countByte on H100 (host Buffer scan)

| Size | H100 Mojo SIMD | H100 Mojo GPU |
|------|----------------|---------------|
| 1MB | **60.1×** | 9.2× |
| 17MB | **89.6×** | 12.2× |
| 105MB | **35.1×** | 6.0× |

AVX-512 byte scanning on Xeon Sapphire Rapids is brutal for this workload — 89.6× at 17MB is the most dramatic CPU-SIMD speedup anywhere in the project, and the H100 GPU trails by 7×. The 1KB row (GPU 0.0×, 25.7 μs per call) is pure CUDA API overhead: cached `DeviceContext` can't get below the kernel launch + device allocation cost, and CPU SIMD completes the same count in 357 ns.

### What would actually make H100 win

1. **Persistent device-resident buffers.** Upload a dataset once, scan it a thousand times. PCIe amortizes away and HBM bandwidth dominates.
2. **Batched N-API API.** Process N inputs per call with async copy-compute overlap so the GPU is never idle waiting for PCIe.
3. **Higher arithmetic intensity.** Matmul (O(n³) compute on O(n²) data), convolution, attention — anything where the compute cost dominates the transfer cost. Stats, grayscale, and countByte are all ≤1 op per byte, the worst possible ratio for any GPU on a PCIe link.

All three are Phase 3 candidate work. See [docs/cloud-benchmark-runbook.md](docs/cloud-benchmark-runbook.md) to reproduce these numbers (~$1, ~30 minutes on RunPod), and the per-addon READMEs for deeper teardowns.

## Phase 3a Cloud Benchmark Results — persistent buffers flipped the result

Phase 3a shipped a new `search_cached.node` addon with a handle-based API that uploads a buffer to device memory once and reuses it across many queries. Same kernel, same benchmark data, new API shape. **It works** — and the magnitude of the win was bigger than predicted.

### countByte on H100, 4-path comparison

| Size | CPU SIMD | GPU one-shot | **GPU cached** | Cached per-call |
|------|---------:|-------------:|---------------:|----------------:|
| 1 MB   | 20.9× |  9.2× |      51.0× |  ~18 μs |
| 17 MB  | 42.9× | 11.3× | **527.8×** |  ~29 μs |
| 105 MB | 33.9× |  5.3× | **1030.7×** | ~94 μs |

*Speedups relative to V8 JS on the same H100 host. Validated 2026-04-11 on RunPod, H100 80GB HBM3.*

**At 17 MB, GPU cached beats CPU SIMD by 12×. At 105 MB, GPU cached beats CPU SIMD by 30×.** This is the first time anywhere in the project that Mojo GPU has beaten Mojo CPU SIMD on a byte-scan benchmark, and the kernel itself is unchanged from the Phase 2c one-shot version. The entire improvement came from changing the API shape: upload once, query many times.

### Why it worked

- **105 MB in 94 μs = 1.1 TB/s effective bandwidth** — roughly 37% of the H100's 3 TB/s HBM3 peak. Excellent efficiency for a byte-scan kernel with shared-memory tree reduction.
- **PCIe is no longer on the hot path.** `loadGpu(105MB)` pays the ~17 ms PCIe transfer exactly once; every subsequent `countByteHandle` call stays entirely on the GPU, reading from device memory at HBM3 speed. Break-even on the upload is reached after a **single reuse**.
- **CPU SIMD is still bounded by host DRAM at ~42 GB/s effective.** HBM3 is ~70× faster once you actually use it. On the one-shot API you can't — on the cached API you can.

### Phase 2d's diagnosis was correct

The Phase 2d H100 run was valuable precisely *because* it was "negative": it identified PCIe as the bottleneck instead of compute, and that diagnosis is what made Phase 3a targeted rather than speculative. The one-shot `countByteGpu` still loses to CPU SIMD on H100 — that hasn't changed. What Phase 3a added is a parallel handle-based API for the workload shape where the loss doesn't have to happen.

See [examples/simd-search/README.md](examples/simd-search/README.md) for the full API, 5-size table with `loadGpu` upload costs, break-even analysis, and the "when to use the handle API" decision rule. Phase 3b (extend the pattern to stats and image) and Phase 3c (tensor-core matmul) are now justified by the 3a result.

## Phase 3b Cloud Benchmark Results — persistent buffers ported to grayscale and stats

Phase 3b tested whether the Phase 3a persistent-buffer template generalizes beyond reductions. Two new cached addons:

- [`image_cached.node`](examples/image/addon_cached.mojo) — `loadImageGpu` + `grayscaleHandle(h, dst)` + `releaseImageGpu`. Transform kernel shape (output same size as input).
- [`stats_cached.node`](examples/stats/addon_cached.mojo) — `loadStatsGpu` + `statsHandle(h)` + `releaseStatsGpu`. Two reduction passes, Float64 input, multi-field result, CPU-side percentiles.

Both shipped with byte-exact correctness against the existing CPU paths (210 + 208 regression cases respectively) and run without crashes on M4. **H100 validation run**: 2026-04-11 on an H100 80GB HBM3 SXM5 via RunPod, same pod configuration as the Phase 3a run.

### Phase 3a search_cached reproduces on SXM (sanity check)

| Size   | CPU SIMD | GPU one-shot | **GPU cached** (PCIe, 3a) | **GPU cached** (SXM, 3b.3) |
| ------ | -------: | -----------: | ------------------------: | -------------------------: |
| 1 MB   |    62.1× |         9.2× |                     51.0× |                      53.5× |
| 17 MB  |    93.4× |        12.4× |                    527.8× |                  **562.9×** |
| 105 MB |    46.5× |         6.3× |                   1030.7× | **1146.4×** |

SXM is +6–11% better than PCIe at the top sizes. Phase 3a reproduces cleanly.

### Phase 3b.1 grayscale cached on H100

| Resolution     | JS   | CPU SIMD | GPU one-shot | **GPU cached** | `loadImageGpu` |
| -------------- | ---- | -------- | ------------ | -------------- | -------------- |
| 720p (3.7 MB)  | 1.0× | 5.6×     | 2.0×         | **11.9×**      | 0.3 ms         |
| 1080p (8.3 MB) | 1.0× | 6.4×     | 2.0×         | **12.6×**      | 0.8 ms         |
| 4K (33.2 MB)   | 1.0× | 6.3×     | 1.7×         | **7.4×**       | 4.9 ms         |

**Result: template works, absolute speedup is small.** Cached beats GPU one-shot 4–6× at every resolution and beats CPU SIMD 1.2–2.1×. Break-even vs one-shot is 1 iteration. But the headline number at 4K (7.4× JS) is well below the Phase 3a countByte 105 MB result (1146× JS) — not because the template failed, but because grayscale is a transform with a per-call 33 MB D2H that can't be amortized. The strategy doc's risk #1 flagged this before the run: "grayscale output is the same size as the input... every `grayscaleHandle` call still pays ~3 ms of D2H at 4K. Cached GPU should beat CPU SIMD but by a much smaller margin than countByte — maybe 3-5× instead of 30×." Observed: 1.2× at 4K. Directionally correct, below the predicted ceiling.

See [examples/image/README.md](examples/image/README.md#phase-3b1--cached-grayscale-api-nvidia-h100-80gb-hbm3) for the full teardown.

### Phase 3b.2 stats cached on H100

| Size | JS   | CPU SIMD | GPU one-shot | **GPU cached** | `loadStatsGpu` |
| ---- | ---- | -------- | ------------ | -------------- | -------------- |
| 100K | 1.0× | 8.4×     | 7.0×         | **8.1×**       | 0.1 ms         |
| 1M   | 1.0× | 7.5×     | 7.1×         | **8.1×**       | 1.0 ms         |
| 10M  | 1.0× | 3.6×     | 3.4×         | **3.5×**       | 13.1 ms        |

**Result: template works, percentile quickselect dominates at 10M.** The cached API replaces the per-call scalar Float64→Float32 cast with a vectorized one-time cast and eliminates per-call PCIe upload and device allocation. At 100K–1M, cached ties or slightly beats CPU SIMD. At 10M the CPU-side percentile quickselect (p50/p95/p99 on 10M Float64) swamps everything at ~200–300 ms per call, so cached saves ~15 ms of GPU-related work per call but the wall clock is dominated by percentiles — giving a 3.5× JS result that's effectively measuring quickselect, not the cached GPU reduction. Note how the JS→CPU-SIMD ratio itself drops from 7.5× at 1M to 3.6× at 10M for the same reason.

See [examples/stats/README.md](examples/stats/README.md#phase-3b2--cached-stats-api-nvidia-h100-80gb-hbm3) for the full teardown.

### What Phase 3b validated and what it didn't

**Validated**: the persistent-buffer template ports to two additional kernel shapes (elementwise transform, two-pass Float64 reduction) without structural changes beyond copying ~150 lines of handle plumbing per addon. Both addons pass byte-exact correctness against the existing CPU SIMD oracles.

**Did not validate**: a 100×-or-better speedup for either kernel. The strategy doc's ≥100× JS / ≥5× CPU SIMD Green decision gates are Red at 4K grayscale (physics: D2H floor) and Red at 10M stats (CPU percentile dominance). Neither is a template failure — both are workload-shape bottlenecks that persistent buffers can't address.

**What the pattern is actually good for**: kernels where (a) output is small relative to input (reductions) and (b) the same input is queried many times. countByte hits both and wins 1000×. Grayscale hits (b) but not (a) and wins 1.2–12× depending on resolution. Stats hits (a) and (b) at small sizes but pays CPU-side percentile cost at large sizes. Matmul (Phase 3c target) hits (a) and (b) *and* has high arithmetic intensity — the prediction is that matmul will be the phase 3 headline result, not these two.

## Phase 3c Cloud Benchmark Results — tensor-core matmul via `linalg.matmul`

Phase 3c shipped a new [`matmul_cached.node`](examples/matmul/addon_cached.mojo) addon that wraps MAX's production [`linalg.matmul`](https://github.com/modular/modular/tree/main/max/kernels/src/linalg) kernel with the Phase 3a persistent-buffer handle API. Upload A and B to the GPU once via `loadMatrixGpu`, then run `matmulHandle(hA, hB, dst)` many times. The kernel body is **5 lines** — no hand-rolled tensor-core code — because `linalg.matmul[target="gpu"](C, A, B, Optional(ctx))` handles tile scheduling, shared memory, swizzle patterns, and tensor-core dispatch internally.

### matmul on H100 (FP32 inputs → TF32 tensor cores)

| Size  | JS       | Mojo CPU parallel | **GPU cached**       |
| ----- | -------- | ----------------- | -------------------- |
| 256²  | 32.3 ms  | 1.00 ms (32×)     | **0.05 ms (606×)**   |
| 512²  | 287 ms   | 13.0 ms (22×)     | **0.13 ms (2166×)**  |
| 1024² | 5750 ms  | 62.0 ms (93×)     | **0.48 ms (12038×)** |
| 2048² | 59591 ms | 561 ms (106×)     | **2.10 ms (28343×)** |

*Speedups relative to single-threaded V8 on the same H100 host. Validated 2026-04-12 on RunPod H100 80GB HBM3.*

**28343× JS at 2048²** — the largest speedup anywhere in this project by 25×, and **267× faster than Mojo CPU parallel** (AVX-512 on Xeon Sapphire Rapids) on the same workload. A single H100 runs ~476 fresh 2048×2048 matmuls per second through the N-API boundary.

### Why this is the phase 3 headline

- **Arithmetic intensity matters**: matmul is O(n³) compute on O(n²) data. At 2048² that's 17 GFLOPs of math on 16 MB of I/O — roughly 1000 flops per byte, the ideal workload for HBM + tensor cores. Compare to Phase 2's grayscale (1 flop per byte) and countByte (0.5 flops per byte) where PCIe dominated.
- **TF32 tensor cores carry the compute**: ~494 TFLOPS peak, ~250 TFLOPS realized. Non-tensor-core FP32 is ~60 TFLOPS — an 8× hardware advantage that shows up immediately when we call the production kernel.
- **Persistent buffers flip the PCIe bottleneck**, same mechanism that worked in Phase 3a. At 2048² each `matmulHandle` call pays ~0.7 ms kernel + ~1.4 ms D2H. A and B stay resident.
- **`linalg.matmul` is a production kernel**: installing the `max` package (one-line pixi.toml change from `mojo`) unlocks the full MAX kernel library. We get cuBLAS-equivalent performance without hand-rolling any tensor-core code.

### Precision caveat: TF32, not FP32-strict

On H100, FP32 inputs run through tensor cores in TF32 (10-bit mantissa, same 8-bit exponent). Per-multiply relative error is ~1e-3, compounding to ~K × 1e-3 for a K-sum matmul — about 6% worst case at K=2048. The regression test uses `rtol=1e-1` + 1% outlier budget to accommodate this. For FP32-strict results, use the CPU `matmulParallel` path in `matmul.node`. This is standard industry behavior, not a Mojo issue — the same tradeoff exists in cuBLAS, PyTorch, and every other framework that uses tensor cores on FP32 inputs.

### What Phase 3 validated, end-to-end

| Phase | Kernel shape | Input size | JS speedup | CPU SIMD speedup | Verdict |
|---|---|---|---:|---:|---|
| 3a   | byte scan reduction     | 105 MB  | 1146× | 34× | Flagship (until 3c) |
| 3b.1 | elementwise transform   | 4K RGBA | 7.4× | 1.2× | Red (D2H floor) |
| 3b.2 | Float64 two-pass reduction | 10M  | 3.5× | 0.97× | Red (CPU percentile) |
| **3c** | **FP32 matmul (TF32 tensor core)** | **2048²** | **28343×** | **267×** | **New flagship** |

The persistent-buffer template generalizes across all four kernel shapes. What changes is the *ceiling*: PCIe sets it for transforms (3b.1), the CPU quickselect for stats (3b.2), and tensor-core arithmetic intensity lets matmul blow past both (3c). For cached GPU addons, **arithmetic intensity is the single biggest determinant of headline speedup**.

See [examples/matmul/README.md](examples/matmul/README.md#phase-3c2--cached-gpu-matmul-via-linalgmatmul-nvidia-h100-80gb-hbm3) for the full teardown and [docs/cloud-benchmark-runbook.md](docs/cloud-benchmark-runbook.md) for reproduction.

## Phase 3d Cloud Benchmark Results — RAG-shape matmul at recall=1.0

Phase 3c validated tensor-core matmul on square shapes; Phase 3d measures it on tall-skinny RAG workloads (`[B, 384] × [384, N]` for semantic search) against real MS-MARCO passage embeddings. Recall is 1.0 by construction — the matmul is exact cosine against a GPU-resident corpus.

```bash
node examples/matmul/matmul_rag.js --fixture=msmarco-10k --full
```

**Headline (H100 80GB HBM3, real MiniLM-L6-v2 embeddings, k=10):**

| Baseline | Latency | Recall@10 | |
|---|---:|---:|---|
| HNSW `ef=100` | 0.20 ms | 1.00 | hnswlib-node |
| HNSW `ef=2000` | 1.79 ms | 1.00 | |
| ORT CPU | 0.13 ms | 1.00 | onnxruntime-node MatMul |
| **GPU `searchHandle`** | **0.06 ms** | **1.00** | Fused matmul + per-row top-k |

3.3–29× faster than HNSW across all recall levels, with guaranteed exact recall. The "ANN tradeoff" evaporates when you have a GPU and a batch. See [docs/writeup-phase3d.md](docs/writeup-phase3d.md) for the full post and [docs/bench-rag-3d-h100-msmarco.txt](docs/bench-rag-3d-h100-msmarco.txt) for the raw capture.

### Package: `@qkstat/rag`

Phase 3d primitives live in [`packages/rag/`](packages/rag/) as a sibling Node.js package — `v0.1.0-pre`, distributed separately from the root examples. Four GPU primitives (`loadMatrixGpu`, `matmulHandle`, `searchHandle`, `releaseMatrixGpu`) plus a thin `GpuIndex` helper. See [`packages/rag/README.md`](packages/rag/README.md). The dynamic-library dependency analysis behind the package's distribution plan is documented in [`docs/mojo-runtime-isolation-spike-findings.md`](docs/mojo-runtime-isolation-spike-findings.md).

## `@qkstat/embed` — local GPU `embed + search` from Node

MiniLM-L6-v2 embeddings on H100 via MAX from inside a Node.js N-API addon, composable with `packages/rag`'s search path. Productized from a 2-day spike (GO verdict 2026-04-17); lives at [`packages/embed/`](packages/embed/).

**MS-MARCO 10k on H100 80GB HBM3** — real passage embeddings (mean-pooled, L2-normalized MiniLM-L6-v2) via [`packages/embed/bench.js`](packages/embed/bench.js), capture at [`docs/bench-post-spike-h100-20260417T020804Z.txt`](docs/bench-post-spike-h100-20260417T020804Z.txt):

| Corpus | Batch | p50 tok + embed + search | p95 | p99 | Search alone (p50) |
|---|---:|---:|---:|---:|---:|
| 1k | 1 | **1.36 ms** | 1.65 | 1.89 | 0.08 ms |
| 10k | 1 | **1.64 ms** | 2.01 | 2.13 | 0.13 ms |
| 10k | 8 | **3.07 ms** | 3.43 | 4.96 | 0.74 ms |
| 10k | 64 | 8.15 ms | 9.75 | 10.45 | 4.18 ms |

Corpus embed throughput (warm, batch-64): **5,137 docs/sec** at 10k. Correctness vs. the `@huggingface/transformers` CPU reference: **0.999990 min cosine** across 100 sanity sentences. Recall is 1.0 by construction — `@qkstat/rag`'s `searchHandle` is exact (fused matmul + per-row top-k), no ANN tradeoff.

The two addons compose in one Node process with separate CUDA contexts — the "kernel-factory" thesis from the portfolio plan. Async variants (`matmulHandleAsync` / `searchHandleAsync` / `embedTokensAsync`) keep the event loop responsive under concurrent load (p99 event-loop jitter 1.43 ms on M4 Metal, 0 ms on H100 — below sampling floor — at 100 concurrent queries).

See [`docs/embedding-kernel-spike-findings.md`](docs/embedding-kernel-spike-findings.md) for the day-by-day execution log.

## Pod-side infrastructure

Scripts supporting H100 cloud benchmark runs live in [`scripts/`](scripts/):

- [`scripts/runpod-launch.sh`](scripts/runpod-launch.sh) — one-shot RunPod launcher with `trap EXIT` termination safety net. Launches a pod, SSHes in, runs your command, captures output, terminates. Works with a persistent Network Volume to cache pixi env + model weights.
- [`scripts/lambda-bench.sh`](scripts/lambda-bench.sh) — Lambda Cloud sibling for the same orchestration. Unused today (capacity issues during the Phase 3d/spike work); kept as fallback.
- [`scripts/bootstrap.sh`](scripts/bootstrap.sh) — canonical pod-side session bootstrap (PATH + auth + `git fetch` + repo sync). Copied onto the Network Volume so each pod session starts from a known state in ~30 s.
- [`scripts/runpod-bench-3{b,c,d}.sh`](scripts/) — on-pod bench runners for each phase.

Phase 3d and the spike established the pattern. Total cloud spend across all phase 3 work + spike: under $20.

## When Node.js developers should reach for Mojo

V8's JIT compiler is already fast for scalar code. The matmul example shows this clearly: Mojo with the *same algorithm* is only 1.9-2.2x faster. A native addon has real costs -- build toolchain, N-API call overhead, platform-specific binaries. Mojo is worth reaching for when:

**The data is large and the work is data-parallel.** Speedups scale with input size across every example: wyhash is 3.7x at 1KB but 66x at 16MB. countByte is 19x at 1MB but 68x at 100MB. If your hot loop processes a TypedArray or Buffer with thousands of elements, Mojo's SIMD `vectorize()` can process 2-8 elements per instruction where V8 processes one.

**You need multi-core parallelism.** V8 is single-threaded. Worker threads exist but require serialization overhead. Mojo's `parallelize()` distributes work across cores with zero-copy shared memory. The matmul example jumps from 15x (SIMD only) to 91x (SIMD + parallel) by adding one line.

**The operation can't be expressed in JS.** Byte-level SIMD (XOR + reduce for pattern matching), 128-bit integer arithmetic (wyhash's folded multiply), and fixed-point pixel math all require bit-width control that JavaScript doesn't offer. These aren't just faster -- they're impossible to write in JS at all.

**When NOT to use Mojo:** String manipulation, JSON parsing, I/O-bound work, small payloads where N-API call overhead dominates, or anything V8 already JIT-compiles well. If your function runs in under ~1ms on typical input, the native call overhead likely isn't worth it.

## Prerequisites

- [pixi](https://prefix.dev/docs/pixi/) with Mojo nightly
- Node.js 18+
- **For GPU benchmarks on Apple Silicon:** Xcode with the Metal Toolchain component installed (`sudo xcode-select -s /Applications/Xcode.app/Contents/Developer && xcodebuild -downloadComponent MetalToolchain`). Builds default to `--target-accelerator metal:4` on Darwin arm64; override per-addon with `STATS_ACCEL=""` etc. to build CPU-only.
- **For GPU benchmarks on Linux/NVIDIA:** builds default to `--target-accelerator sm_90` (H100/H200). For other NVIDIA architectures, set `STATS_ACCEL="--target-accelerator sm_80"` (A100), `sm_89` (L40/RTX40), etc.
- **Rented cloud GPU walkthrough:** see [docs/cloud-benchmark-runbook.md](docs/cloud-benchmark-runbook.md) for a ~30-minute RunPod H100 benchmark flow.

## Quick Start

```bash
npm install
pixi install

# Build all examples
npm run build:all

# Run benchmarks
node examples/matmul/matmul.js
node examples/simd-search/search.js
node examples/stats/stats.js
node examples/image/image.js
node examples/wyhash/hash.js
```

## Development

To test examples against a local (unreleased) version of napi-mojo:

```bash
cd /path/to/napi-mojo && npm link
cd /path/to/mojo-addon-examples && npm link napi-mojo
```

This replaces the npm-installed package with a symlink to your local checkout. Run `npm install` to revert back to the published package.

## Architecture

Each example is self-contained:

```text
example-name/
  addon.mojo          # Mojo source (SIMD kernels + N-API callbacks)
  example.js           # Demo + benchmark script
  build.sh             # Build script (compile .mojo -> .node)
  README.md            # Example-specific docs + benchmark results
```

All examples depend on napi-mojo for the N-API framework (`napi.types`, `napi.framework.*`).

### Common Patterns

- **Zero-copy TypedArray access:** `JsTypedArray.data_ptr(env).unsafe_bitcast[Float64]()` reads JS memory directly
- **SIMD vectorize:** `vectorize[simd_width_of[DType.float64]()](size, compute)` with `unified {mut}` closure
- **Multi-core parallel:** `parallelize[worker](num_workers)` with `capturing` closure
- **Runtime init:** `KGEN_CompilerRT_AsyncRT_CreateRuntime` via `OwnedDLHandle()` for parallelize in shared libs
