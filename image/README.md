# Image Processing — Pixel Operations

Four RGBA pixel operations on Uint8Arrays. Integer-approximation grayscale, fixed-point brightness, separable box blur with parallel horizontal + vertical passes.

## Functions

| Export | Description | Mojo Feature |
|--------|-------------|-------------|
| `grayscale(rgba, w, h)` | Luminance conversion | Integer `(77R+150G+29B)>>8`, `parallelize()` |
| `brightness(rgba, w, h, factor)` | Multiply + clamp | Fixed-point arithmetic, `parallelize()` |
| `threshold(rgba, w, h, value)` | Binary black & white | Grayscale + compare, `parallelize()` |
| `blur(rgba, w, h, radius)` | Separable box blur | Sliding window, parallel rows + cols |

All return a new `Uint8Array` and preserve the alpha channel.

## Results (M4 Mac)

| Function | 720p | 1080p | 4K |
|----------|------|-------|-----|
| grayscale | 6.8x | 5.4x | **6.8x** |
| brightness | 4.7x | 5.1x | 5.0x |
| threshold | 5.5x | 5.2x | **6.5x** |
| **blur(r=5)** | 6.8x | **10.1x** | 5.6x |

## Build & Run

```bash
npm run build:image
node image/image.js
```
