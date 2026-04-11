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

**GPU grayscale is slower than the JS baseline on M4** — this is intentionally published. Integrated GPUs with shared memory get no bandwidth benefit from "copying to device" (the copy is pure overhead), and the grayscale kernel is so trivial (~20 integer ops per pixel) that the GPU can't amortize the copy + dispatch cost. Output matches the CPU path byte-for-byte.

At the time this was written we predicted the pattern would flip on a discrete GPU with real HBM. **It did not.** See the H100 results below.

## Results (NVIDIA H100 80GB HBM3, via RunPod)

| Function | 720p | 1080p | 4K |
|----------|------|-------|-----|
| grayscale CPU SIMD | 3.2× | 4.0× | **4.3×** |
| grayscale GPU | 2.6× | 2.1× | 1.4× |

On NVIDIA H100, the CPU SIMD path (Xeon Sapphire Rapids AVX-512) beats the Mojo GPU kernel at every resolution — **and the gap widens with image size**. This contradicts the Phase 2d prediction that "GPU becomes bandwidth-bound instead of copy-bound" on real HBM. The opposite is true: for one-shot elementwise calls, PCIe dominates regardless of HBM bandwidth.

**Why**:
- **4K frame = 33.2 MB input + 33.2 MB output.** The full round trip at PCIe Gen4 x16 (~12 GB/s effective) is ~5.5 ms minimum, before any kernel work. That's most of the 25.8 ms total per-call GPU time.
- **CPU SIMD reaches ~30 GB/s effective DRAM bandwidth** via AVX-512, with no copy step. For a 1-op-per-pixel elementwise kernel, CPU wins every time on single-shot calls.
- **The GPU kernel itself is <1 ms** — the H100 is idle for ~95% of the call, waiting on PCIe and pinned-buffer allocations.

The topology matters more than the bandwidth. A discrete GPU's HBM3 is blazing fast, but you have to get data there first. CPU SIMD sits directly on DRAM and wins any workload where you touch the data once.

**What would make grayscale win on H100**: persistent device-resident frame buffer (video pipeline where frames upload once and are processed many times), or operation fusion where grayscale → threshold → blur chain keeps intermediates on the device.

## Build & Run

```bash
pixi run bash image/build.sh
node image/image.js
```
