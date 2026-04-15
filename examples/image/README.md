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

## Phase 3b.1 — Cached grayscale API (NVIDIA H100 80GB HBM3)

Shipped as a separate [`image_cached.node`](addon_cached.mojo) addon with a handle-based API that uploads an image to the GPU once and reuses both the source and destination device buffers across many `grayscaleHandle` calls:

```js
const cached = require('./build/image_cached.node');
const handle = cached.loadImageGpu(rgba, w, h);   // one-time upload
const dst = new Uint8Array(rgba.length);           // caller-owned, reused
for (let i = 0; i < N; i++) {
  cached.grayscaleHandle(handle, dst);             // fills dst in place
}
cached.releaseImageGpu(handle);
```

Validated 2026-04-11 on H100 80GB HBM3 SXM5 via RunPod. Four-path comparison (JS / CPU SIMD / GPU one-shot / GPU cached):

| Resolution     | JS   | CPU SIMD | GPU one-shot | **GPU cached** | `loadImageGpu` |
| -------------- | ---- | -------- | ------------ | -------------- | -------------- |
| 720p (3.7 MB)  | 1.0× | 5.6×     | 2.0×         | **11.9×**      | 0.3 ms         |
| 1080p (8.3 MB) | 1.0× | 6.4×     | 2.0×         | **12.6×**      | 0.8 ms         |
| 4K (33.2 MB)   | 1.0× | 6.3×     | 1.7×         | **7.4×**       | 4.9 ms         |

Break-even vs one-shot is **1 iteration** at every size — the persistent buffers pay for themselves on the very first reuse.

**What the cached API validates**: the Phase 3a persistent-buffer template (originally shipped for countByte) ports cleanly to a *transform* kernel shape. Cached beats GPU one-shot 4–6× at every resolution and beats CPU SIMD 1.2–2.1×, with correctness byte-exact vs the CPU SIMD path on 210 regression cases at three resolutions × three seeds.

**Why the absolute speedup is small** (7.4× at 4K vs 1030× for countByte at 105 MB): grayscale is a transform, not a reduction. Every `grayscaleHandle` call still pays full D2H for the 33 MB output at 4K — roughly 3 ms at PCIe Gen4 ~12 GB/s. That's an irreducible floor. Amortizing `loadImageGpu` eliminates the H2D leg (and the per-call device allocation), but the D2H leg remains per-call and caps the margin vs CPU SIMD. CPU SIMD does 4K grayscale in 5.05 ms on the H100 host's Xeon; cached GPU does it in 4.29 ms — a 1.18× edge dominated by that unavoidable D2H floor.

Put another way: the cached template *works* and the pattern generalizes, but transforms with large output buffers don't exhibit the dramatic wins that pure reductions do. The right use case for the cached grayscale API is a video pipeline that processes many frames of the *same* RGBA buffer — e.g., comparison against a reference image, or successive transformations on a single input — not a pipeline that streams new frames through.

**Cached test_cached.js on H100 (leak smoke)**: 500 load+release iterations on 720p show ~3.3 MB/iter RSS growth, compared to ~0 MB/iter for the `search_cached` leak smoke on the same host. The extra per-cycle leak is image-specific (not present in the reduction template) and persists on both M4 Metal and H100 CUDA backends — it correlates with the kernel call inside the release cycle, not with the load/release itself. Production usage (load once, query many, release once) does not trigger it; the reused-handle path shows zero growth.

## Build & Run

```bash
pixi run bash image/build.sh
node image/image.js
```
