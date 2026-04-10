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

## Build & Run

```bash
pixi run bash stats/build.sh
node stats/stats.js
```
