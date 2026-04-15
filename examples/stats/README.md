# Statistics — SIMD Aggregation

Compute `{mean, stddev, min, max, p50, p95, p99}` on Float64Arrays in a single N-API call. SIMD reductions + parallel accumulation + quickselect percentiles.

## Functions

| Export | Description | Mojo Feature |
|--------|-------------|-------------|
| `stats(data)` | Full summary stats (CPU) | SIMD reduce_add/min/max, `parallelize()` |
| `statsGpu(data)` | Same, on GPU | Metal reductions via `DeviceContext` + shared-memory tree reduction |
| `histogram(data, bins)` | Bin counts as `Float64Array` | SIMD min/max range detection |

## Results (M4 Mac)

| Function | 100K | 1M | 10M |
|----------|------|-----|-----|
| **stats() CPU** | **4.2x** | **5.8x** | **6.7x** |
| stats() GPU Metal | 2.1x | 4.0x | 4.2x |
| histogram() CPU | 3.7x | 3.9x | 4.0x |

**CPU row** — `vectorize()` + `parallelize()`, Float64 throughout.

**GPU row** — Apple M4 integrated GPU via Mojo's Metal 4 backend (Tier-3 support). Kernels run in **Float32** because Metal compute shaders don't support Float64; data is cast Float64→Float32 on the host→device copy and the final reduction happens back on the host in Float64. Relative error vs the CPU path is <1e-4 on uniform random data. `DeviceContext` is cached across calls via N-API instance data, so per-call overhead is dominated by the H→D copy, not context creation. For adversarial or very wide-dynamic-range inputs, stick with the CPU path.

## Results (NVIDIA H100 80GB HBM3, via RunPod)

| Size | H100 Mojo SIMD | H100 Mojo GPU |
|------|----------------|---------------|
| 10K | 4.3× | 3.5× |
| 100K | 5.0× | 4.7× |
| 1M | 5.3× | 5.2× |
| 10M | **8.3×** | 7.6× |

*Ratios are vs V8 on the same H100 host (Xeon Sapphire Rapids).* **CPU SIMD beats GPU at every size.** This contradicts the Phase 2d prediction in [docs/cloud-benchmark-runbook.md](../docs/cloud-benchmark-runbook.md), which expected stats on H100 to land in the 20-50× range based purely on HBM3 bandwidth reasoning. That reasoning missed the PCIe bottleneck.

**Why the H100 GPU underperforms here:**

1. **Two-pass reduction, each with its own copy cycle.** Stats computes `{sum, min, max}` in one kernel pass and `sum_sq_diff(mean)` in a second pass (because the second needs the mean from the first). Each pass does a fresh pinned-buffer allocation, a host-side Float64→Float32 cast, an H2D copy, a kernel launch, and a D2H copy of partial sums. Two full cycles per `stats()` call.
2. **Float32 cast is a scalar host loop**, not vectorized. At [addon.mojo:327](addon.mojo#L327) and [addon.mojo:388](addon.mojo#L388):
   ```mojo
   for i in range(size):
       host_ptr[i] = Float32(data[i])
   ```
   At 10M elements this loop is tens of ms per call — probably the single largest bottleneck.
3. **PCIe bandwidth caps data transfer at ~12 GB/s.** A 10M Float64 array is 80 MB; H2D takes ~6.7 ms minimum. The H100's 3 TB/s HBM3 compute time for the reduction itself is ~27 μs — so the PCIe transfer is roughly 250× longer than the actual kernel work.

**What would fix it** (Phase 3 candidate work):
- A fused single-pass kernel using Welford's online variance algorithm (one pass instead of two)
- Device-side Float64→Float32 cast kernel (eliminates the host scalar loop)
- Or a full Float64 kernel on NVIDIA (CUDA supports it natively; only Metal is Float32-only)
- Persistent device buffers across calls (upload once, reduce many times)

**The honest read**: single-shot `stats()` on a host-resident Float64Array is bottlenecked by data transfer, not compute, regardless of how fast the GPU's HBM is.

## Phase 3b.2 — Cached stats API (NVIDIA H100 80GB HBM3)

Shipped as a separate [`stats_cached.node`](addon_cached.mojo) addon with a handle-based API that casts the Float64 input to Float32 once (via a vectorized host-side SIMD loop, replacing the scalar cast at [addon.mojo:327](addon.mojo#L327) and [addon.mojo:388](addon.mojo#L388)), uploads the Float32 buffer to the GPU once, and reuses the persistent device buffers + pinned partial-sum buffers across many `statsHandle` calls. The handle also owns a heap Float64 copy so percentile quickselect can run without requiring the caller to pass the original array back in:

```js
const cached = require('./build/stats_cached.node');
const h = cached.loadStatsGpu(data);       // vectorized F64→F32 cast + H2D, one time
for (let i = 0; i < N; i++) {
  const s = cached.statsHandle(h);         // {mean, stddev, min, max, p50, p95, p99}
}
cached.releaseStatsGpu(h);
```

Validated 2026-04-11 on H100 80GB HBM3 SXM5 via RunPod. Four-path comparison (JS / CPU SIMD / GPU one-shot / GPU cached):

| Size | JS   | CPU SIMD | GPU one-shot | **GPU cached** | `loadStatsGpu` |
| ---- | ---- | -------- | ------------ | -------------- | -------------- |
| 100K | 1.0× | 8.4×     | 7.0×         | **8.1×**       | 0.1 ms         |
| 1M   | 1.0× | 7.5×     | 7.1×         | **8.1×**       | 1.0 ms         |
| 10M  | 1.0× | 3.6×     | 3.4×         | **3.5×**       | 13.1 ms        |

Break-even vs one-shot is **1 iteration** at every size. Correctness regression passes 208 cases across 4 sizes × 2 seeds + 200-query stability, all within 1e-4 relative tolerance of the Float64 CPU path.

**What the cached API validates**: the Phase 3a persistent-buffer template ports to the hardest kernel shape in the project — two reduction passes, Float64 input, multi-field result object. At 100K–1M the cached path ties or slightly beats CPU SIMD. The vectorized one-time cast (~10 ms at 10M) replaces the per-call scalar cast (tens of ms per call in the one-shot path) and the persistent Float32 device buffer eliminates per-call PCIe upload + device allocation.

**Why 10M flattens to 3.5× JS**: at 10M elements, CPU-side percentile quickselect dominates per-call cost. Each `statsHandle` call runs three O(n) quickselects (p50, p95, p99) against a scratch copy of the cached Float64 data — roughly 200–300 ms per call on this Xeon. Meanwhile the GPU-related per-call savings from caching are only ~15 ms (370 ms one-shot → 355 ms cached), because the H100 Xeon is fast enough that the Float64→Float32 cast was never as expensive on this host as it was on the Phase 2d CPU baseline. So the `statsHandle` benchmark at 10M is effectively measuring CPU quickselect time, not the cached GPU reduction, and the JS→CPU-SIMD ratio itself collapses from 7.5× at 1M to 3.6× at 10M for the same reason.

**Decision gate verdict**: below the strategy doc's ≥100× JS Green gate at 10M and below the ≥30× Yellow gate — a **Red** outcome. But the Red comes from the percentile step that sits outside the cached GPU path, not from a failure of the cached template. The reduction kernels themselves hit the same per-call wall-clock as the original `statsGpu` path; the cached API just saves the ~15 ms of cast + upload + allocation on each call. Moving percentiles to the GPU (parallel quickselect or radix partition) is the natural next step — explicitly deferred in the Phase 3 strategy doc as "a much harder problem and orthogonal to the bandwidth story."

**Recommended use**: the cached stats API is a meaningful win for 100K–1M element workloads in hot loops, where percentiles are cheap and the cast amortization matters. For the 10M+ case where percentiles dominate, either use the CPU `stats()` path (which is bottlenecked by the same percentile work but avoids the GPU round trip entirely), or wait for a future phase that moves percentile computation to the device.

**Cached test_cached.js on H100 (leak smoke)**: 300 load+release iterations on 100K elements show ~0.7 MB/iter RSS growth — lower than image_cached (~3.3 MB/iter) but still present. Production usage (load once, query many, release once) does not trigger the leak; only synthetic load/release loops do.

## Build & Run

```bash
pixi run bash stats/build.sh
node stats/stats.js
```
