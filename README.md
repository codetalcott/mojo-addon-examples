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

See [ROADMAP.md](ROADMAP.md) for planned examples.

## Prerequisites

- [pixi](https://prefix.dev/docs/pixi/) with Mojo nightly
- Node.js 18+
- napi-mojo (linked via npm)

## Quick Start

```bash
npm install
pixi install
pixi run bash matmul/build.sh
node matmul/matmul.js
```
