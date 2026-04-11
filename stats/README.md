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

## Build & Run

```bash
pixi run bash stats/build.sh
node stats/stats.js
```
