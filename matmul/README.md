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

## Results (M4 Mac)

| Step | 1024x1024 | 2048x2048 |
|------|-----------|-----------|
| JS baseline | 1x | 1x |
| Mojo naive | 2.2x | 1.9x |
| Mojo vectorized | 15.5x | 32.0x |
| Mojo tiled | 12.6x | 27.6x |
| **Mojo parallel** | **38.6x** | **91.4x** |

## Build & Run

```bash
npm run build:matmul
node matmul/matmul.js
```
