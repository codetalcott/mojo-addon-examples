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

### Matrix Multiply (Progressive Optimization)
**Location:** `matmul/`

The canonical Mojo performance demo. Four exported functions showing progressive optimization:

| Step | 1024x1024 | 2048x2048 | Mojo Feature |
|------|-----------|-----------|-------------|
| JS baseline | 1x | 1x | — |
| Mojo naive | 2.2x | 1.9x | Same algorithm, better compiler |
| Mojo vectorized | 15.5x | 32.0x | `vectorize()` — SIMD inner loop |
| Mojo tiled | 12.6x | 27.6x | Cache-friendly 64x64 blocking |
| **Mojo parallel** | **38.6x** | **91.4x** | `parallelize()` — multi-core |

### SIMD Text Search
**Location:** `simd-search/`

Byte-level SIMD pattern matching — impossible in pure JavaScript. XOR-based SIMD byte matching with `parallelize()` for large buffers.

| Function | 1MB | 100MB | Technique |
|----------|-----|-------|-----------|
| **countByte** | **19.2x** | **67.6x** | SIMD XOR + reduce + parallelize |
| countLines | 18.5x | 65.3x | Same kernel, byte=0x0A |
| searchAll | 2.5x | 3.1x | Two-pass: SIMD count + collect |

### Statistics / Aggregation
**Location:** `stats/`

Compute `{mean, stddev, min, max, p50, p95, p99}` on a Float64Array in a single N-API call. SIMD reductions + parallel accumulation + quickselect percentiles.

| Function | 1M | 10M | Technique |
|----------|-----|-----|-----------|
| **stats()** | **5.8x** | **6.7x** | SIMD reduce_add/min/max + parallelize |
| histogram() | 3.9x | 4.0x | SIMD min/max range detection |

### Image Processing (Pixel Operations)
**Location:** `image/`

Four RGBA pixel operations: grayscale, brightness, threshold, separable box blur. All parallelized across rows/columns.

| Function | 720p | 1080p | 4K | Technique |
|----------|------|-------|-----|-----------|
| grayscale | 6.8x | 5.4x | **6.8x** | Integer `(77R+150G+29B)>>8` + parallelize |
| brightness | 4.7x | 5.1x | 5.0x | Fixed-point multiply + clamp + parallelize |
| threshold | 5.5x | 5.2x | **6.5x** | Grayscale + compare + parallelize |
| **blur(r=5)** | 6.8x | **10.1x** | 5.6x | Separable box blur, parallel rows + cols |

---

## Planned — Niche / Higher Effort

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
