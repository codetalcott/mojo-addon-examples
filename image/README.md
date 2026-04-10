# Image Processing — Pixel Operations

Four RGBA pixel operations on Uint8Arrays. Integer-approximation grayscale, fixed-point brightness, separable box blur with parallel horizontal + vertical passes.

## Functions

| Export | Description | Mojo Feature |
|--------|-------------|-------------|
| `grayscale(rgba, w, h)` | Luminance conversion (CPU) | Integer `(77R+150G+29B)>>8`, `parallelize()` |
| `grayscaleGpu(rgba, w, h)` | Same, on GPU | Metal elementwise kernel, one thread per pixel |
| `brightness(rgba, w, h, factor)` | Multiply + clamp | Fixed-point arithmetic, `parallelize()` |
| `threshold(rgba, w, h, value)` | Binary black & white | Grayscale + compare, `parallelize()` |
| `blur(rgba, w, h, radius)` | Separable box blur | Sliding window, parallel rows + cols |

All return a new `Uint8Array` and preserve the alpha channel.

## Results (M4 Mac)

| Function | 720p | 1080p | 4K |
|----------|------|-------|-----|
| grayscale CPU | 6.8x | 5.4x | **6.8x** |
| grayscale GPU Metal | 0.6x | 0.6x | 0.8x |
| brightness | 4.7x | 5.1x | 5.0x |
| threshold | 5.5x | 5.2x | **6.5x** |
| **blur(r=5)** | 6.8x | **10.1x** | 5.6x |

**GPU grayscale is slower than the JS baseline on M4** — this is intentionally published. Integrated GPUs with shared memory get no bandwidth benefit from "copying to device" (the copy is pure overhead), and the grayscale kernel is so trivial (~20 integer ops per pixel) that the GPU can't amortize the copy + dispatch cost. Output matches the CPU path byte-for-byte. This data point is expected to flip on a discrete GPU with real HBM (e.g. H100, MI300X) where the kernel becomes bandwidth-bound instead of copy-bound. See Phase 2d cloud benchmark notes.

## Build & Run

```bash
pixi run bash image/build.sh
node image/image.js
```
