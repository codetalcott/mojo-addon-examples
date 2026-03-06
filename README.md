# Mojo Addon Examples

High-performance Node.js addon examples built with [napi-mojo](https://github.com/codetalcott/mojo-node-api). Each example demonstrates Mojo's SIMD `vectorize()` and `parallelize()` through the N-API bridge, with benchmarks against pure JavaScript.

## Examples

### Matrix Multiply — Progressive Optimization

Four implementations showing Mojo's optimization story, from naive triple loop to SIMD + tiled + parallel:

```
node matmul/matmul.js
```

**Results (M4 Mac, Float64):**

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
node simd-search/search.js
```

**Results (M4 Mac, Buffer/Uint8Array):**

| Function | 1MB | 16MB | 100MB | Mojo Feature |
|----------|-----|------|-------|-------------|
| **countByte** | **19.2x** | **52.4x** | **67.6x** | SIMD XOR + reduce, `parallelize()` |
| countLines | 18.5x | 50.1x | 65.3x | Same kernel, byte=0x0A |
| searchAll (1-byte) | 2.5x | 2.8x | 3.1x | Two-pass: SIMD count + collect |
| searchAll (multi-byte) | 2.0x | 2.3x | 2.5x | First+last byte SIMD filter |

### Statistics — SIMD Aggregation

Compute `{mean, stddev, min, max, p50, p95, p99}` on Float64Arrays in a single call. SIMD reductions + parallel accumulation + quickselect percentiles.

```
node stats/stats.js
```

**Results (M4 Mac, Float64):**

| Function | 100K | 1M | 10M | Mojo Feature |
|----------|------|-----|-----|-------------|
| **stats()** | **4.2x** | **5.8x** | **6.7x** | SIMD reduce_add/min/max, `parallelize()` |
| histogram() | 3.7x | 3.9x | 4.0x | SIMD min/max range detection |

### Image Processing — Pixel Operations

Four RGBA pixel operations on Uint8Arrays: `grayscale`, `brightness`, `threshold`, `blur`. Integer-approximation grayscale, fixed-point brightness, separable box blur with parallel horizontal + vertical passes.

```
node image/image.js
```

**Results (M4 Mac, RGBA Uint8Array):**

| Function | 720p | 1080p | 4K | Mojo Feature |
|----------|------|-------|-----|-------------|
| grayscale | 6.8x | 5.4x | **6.8x** | Integer `(77R+150G+29B)>>8`, `parallelize()` |
| brightness | 4.7x | 5.1x | 5.0x | Fixed-point multiply + clamp, `parallelize()` |
| threshold | 5.5x | 5.2x | **6.5x** | Grayscale + compare, `parallelize()` |
| **blur(r=5)** | 6.8x | **10.1x** | 5.6x | Separable box blur, parallel rows + cols |

See [ROADMAP.md](ROADMAP.md) for planned examples.

## Prerequisites

- [pixi](https://prefix.dev/docs/pixi/) with Mojo nightly
- Node.js 18+
- napi-mojo (linked via npm)

## Quick Start

```bash
npm install
pixi install

# Build all examples
npm run build:matmul
npm run build:search
npm run build:stats
npm run build:image

# Run benchmarks
node matmul/matmul.js
node simd-search/search.js
node stats/stats.js
node image/image.js
```
