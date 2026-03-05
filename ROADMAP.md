# Mojo Addon Examples — Roadmap

High-performance Node.js addon examples built with [napi-mojo](https://github.com/codetalcott/mojo-node-api). Each example demonstrates Mojo's SIMD `vectorize()` and `parallelize()` through the N-API bridge, with benchmarks against pure JavaScript and V8 JIT.

## Completed

### Vector Similarity (in napi-mojo repo)
**Location:** `mojo-node-api/examples/vectors-addon.mojo`

Dot product, cosine similarity, euclidean distance on Float64Arrays. SIMD + parallel across CPU cores for large vectors.

| Dimension | Mojo SIMD+Parallel | V8 JIT | Speedup |
|-----------|-------------------|--------|---------|
| 10,000 | 181K ops/sec | 55K ops/sec | **3.3x** |
| 100,000 | 50K ops/sec | 5.5K ops/sec | **9.1x** |

---

## Tier 1 — High Impact, Low Effort

### 1. Matrix Multiply (Progressive Optimization)
**Status:** In progress
**Effort:** 1-2 days | **Expected speedup:** 50-100x over JS

The canonical Mojo performance demo. Four exported functions showing progressive optimization:

| Step | Function | Mojo Feature | Expected Gain |
|------|----------|-------------|---------------|
| 1 | `matmulNaive` | Triple loop baseline | 1x |
| 2 | `matmulVectorized` | `vectorize()` inner loop | 2-4x |
| 3 | `matmulTiled` | Cache-friendly blocking | 10-20x |
| 4 | `matmulParallel` | `parallelize()` across tile rows | 30-100x |

Benchmark at 128, 256, 512, 1024, 2048 dimensions. Each step maps to exactly one language feature. The most effective demo for explaining *why Mojo exists*.

### 2. SIMD Text Search
**Status:** Planned
**Effort:** 1-2 days | **Expected speedup:** 5-15x

Byte-level SIMD pattern matching — impossible in pure JavaScript. Operations:

- `countByte(buffer, byte)` — SIMD `cmp_eq` + `reduce_add` (simdjson technique)
- `countLines(buffer)` — count newlines at SIMD speed
- `searchAll(buffer, needle)` — return Uint32Array of all match positions

Processes 16 bytes/cycle (ARM NEON) or 32 bytes/cycle (AVX2). Guaranteed massive speedup on 1MB+ buffers. Zero UTF-8 complications (byte-level search is safe for ASCII patterns in UTF-8).

---

## Tier 2 — High Impact, Moderate Effort

### 3. Statistics / Aggregation
**Status:** Planned
**Effort:** 2-3 days | **Expected speedup:** 10x

Compute `{mean, stddev, min, max, p50, p95, p99}` on a Float64Array in a single N-API call. SIMD reductions + parallel accumulation. The most *practical* example — every monitoring, observability, and analytics pipeline needs this.

- `stats(data)` — full stats object in one pass
- `histogram(data, bins)` — SIMD-accelerated binning
- `topK(data, k)` — partial sort via quickselect

### 4. Image Processing (Pixel Operations)
**Status:** Planned
**Effort:** 2-3 days | **Expected speedup:** 3-10x on pixel ops

SIMD + parallel pixel processing on raw RGBA buffers. Decode/encode via sharp (JS), pixel math in Mojo.

- `grayscale(rgba, width, height)` — SIMD weighted channel reduction
- `blur(rgba, width, height, radius)` — separable box blur, parallel across rows
- `brightness(rgba, width, height, factor)` — trivially SIMD-parallel
- `threshold(rgba, width, height, value)` — SIMD comparison

Visual before/after demo on 4K images.

---

## Tier 3 — Niche / Higher Effort

### 5. Audio DSP (FFT + Convolution)
**Effort:** 5-8 days | Radix-2 FFT butterfly ops map perfectly to SIMD multiply-add. Visual demo with waveform-to-spectrum conversion.

### 6. Distance Metrics (Hamming + Levenshtein)
**Effort:** 5-8 days | Myers' bit-vector Levenshtein processes 64 characters per op. Useful for fuzzy search and deduplication.

### 7. SIMD Hash (wyhash)
**Effort:** 3-5 days | Developer experience story — match C performance in 50 lines of Mojo. First native wyhash addon for Node.js.

---

## Architecture

Each example is self-contained:

```
example-name/
  addon.mojo          # Mojo source (SIMD kernels + N-API callbacks)
  example.js           # Demo + benchmark script
  build.sh             # Build script (compile .mojo -> .node)
  README.md            # Example-specific docs + benchmark results
```

All examples depend on napi-mojo for the N-API framework (`napi.types`, `napi.framework.*`).

### Common Patterns

- **Zero-copy TypedArray access:** `JsTypedArray.data_ptr(env).bitcast[Float64]()` reads JS memory directly
- **SIMD vectorize:** `vectorize[simd_width_of[DType.float64]()](size, compute)` with `unified {mut}` closure
- **Multi-core parallel:** `parallelize[worker](num_workers)` with `capturing` closure
- **Runtime init:** `KGEN_CompilerRT_AsyncRT_CreateRuntime` via `OwnedDLHandle()` for parallelize in shared libs

### Build

```bash
pixi run bash example-name/build.sh
node example-name/example.js
```
