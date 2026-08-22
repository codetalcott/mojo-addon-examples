# Statistics — SIMD Aggregation

Compute `{mean, stddev, min, max, p50, p95, p99}` on Float64Arrays in a single N-API call. SIMD reductions + parallel accumulation + quickselect percentiles.

## Functions

| Export | Description | Mojo Feature |
|--------|-------------|-------------|
| `stats(data)` | Full summary stats | SIMD reduce_add/min/max, `parallelize()` |
| `histogram(data, bins)` | Bin counts as `Float64Array` | SIMD min/max range detection |

## Results (M4 Mac)

| Function | 100K | 1M | 10M |
|----------|------|-----|-----|
| **stats()** | **4.2x** | **5.8x** | **6.7x** |
| histogram() | 3.7x | 3.9x | 4.0x |

## Build & Run

```bash
npm run build:stats
node stats/stats.js
```
