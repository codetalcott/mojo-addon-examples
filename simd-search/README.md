# SIMD Text Search — Byte-Level Pattern Matching

SIMD byte scanning that's impossible to express in pure JavaScript. XOR-based byte matching with `parallelize()` for large buffers.

## Functions

| Export | Description | Mojo Feature |
|--------|-------------|-------------|
| `countByte(buf, byte)` | Count occurrences of a byte (CPU) | SIMD XOR + reduce, `parallelize()` |
| `countByteGpu(buf, byte)` | Same, on GPU | Metal tree reduction, shared memory partials |
| `countLines(buf)` | Count newlines (`wc -l`) | Same kernel as countByte, byte=0x0A |
| `searchAll(buf, needle)` | All match positions as `Uint32Array` | Two-pass: SIMD count + collect |

Accepts both `Buffer` and `Uint8Array` inputs. Multi-byte `searchAll` uses a first+last byte SIMD filter to minimize full comparisons.

## Results (M4 Mac)

| Function | 1MB | 16MB | 100MB |
|----------|-----|------|-------|
| **countByte CPU** | **19.2x** | **52.4x** | **67.6x** |
| countByte GPU Metal | 0.9x | 1.0x | 2.3x |
| countLines | 18.5x | 50.1x | 65.3x |
| searchAll (1-byte) | 2.5x | 2.8x | 3.1x |
| searchAll (multi-byte) | 2.0x | 2.3x | 2.5x |

**GPU countByte is dramatically slower than CPU SIMD on M4** — at 100MB it's 2.3× JS vs CPU SIMD's 67.6×, a **30× gap**. This is the worst-case shape for integrated-GPU computation: the kernel is trivial (one compare + increment per byte), the data is huge (H2D copy dominates), and the CPU side already has fast SIMD + multi-core parallelism over direct DRAM. The GPU result is correct — output matches CPU byte-count exactly — but the M4 doesn't have the bandwidth headroom to amortize the dispatch cost on a byte-scan kernel.

At the time this was written we predicted the pattern would flip on H100 (3TB/s HBM) where GPU would become "compute-bound on the reduction rather than copy-bound." **It did not.** See the H100 results below.

## Results (NVIDIA H100 80GB HBM3, via RunPod)

| Size | H100 Mojo SIMD | H100 Mojo GPU |
|------|----------------|---------------|
| 1MB | **60.1×** | 9.2× |
| 17MB | **89.6×** | 12.2× |
| 105MB | **35.1×** | 6.0× |

**countByte on H100 is dramatically slower than Mojo CPU SIMD at every size.** The 89.6× CPU SIMD number at 17MB is the most dramatic speedup anywhere in the project — Xeon Sapphire Rapids AVX-512 byte-scanning outperforms the H100 GPU path by roughly **7×**. The prediction that GPU would win on discrete HBM hardware was wrong.

**Why GPU loses at byte scanning on a PCIe-attached accelerator**:

- **~1 arithmetic op per byte is the worst possible ratio for any GPU on a PCIe link.** At 105 MB you pay ~8.7 ms just for the H2D copy (PCIe Gen4 x16 at ~12 GB/s effective). The H100's kernel compute time on the same 105 MB at 3 TB/s HBM3 is ~35 μs. **Data transfer is ~250× longer than actual compute.**
- **AVX-512 on Xeon hits ~30 GB/s effective DRAM bandwidth for byte scans** because the data is already on the CPU's memory controller. No copy step, no PCIe, no kernel launch. CPU wins by an order of magnitude.
- **The 1KB row (GPU 0.0×, 25.7 μs per call)** is pure CUDA API overhead: kernel launch + device allocation + free, even with cached `DeviceContext`. CPU SIMD does the same count in 357 ns.

**The honest takeaway**: byte-scan + reduction is the worst possible workload shape for single-shot N-API GPU calls, on any hardware. The H100's HBM3 is wasted because the data never lives on the device long enough to benefit from it.

**What would make H100 win byte scanning** (Phase 3 candidate): a persistent device-resident buffer API (upload a dataset once, query it N times), or batched countByte that processes multiple queries in a single kernel launch against already-loaded data.

## Phase 3a: Persistent Device Buffers (validated on H100, 2026-04-11)

Phase 3a.1 shipped a prototype `search_cached.node` exposing a new handle-based API that uploads a buffer to device memory once and reuses it across many queries, amortizing PCIe transfer and per-call allocation. **The hypothesis — that this flips the H100 result — validated dramatically**.

### API

```js
const cached = require('./build/search_cached.node');

const buf = fs.readFileSync('big-log.txt');     // e.g. 100 MB
const h = cached.loadGpu(buf);                  // one-shot upload to device

// Many queries against the resident buffer, no per-call H2D copy
for (const byte of [0x0A, 0x41, 0x2E, 0x20]) {
  const n = cached.countByteHandle(h, byte);    // ~94 μs each at 105 MB
  console.log(byte, n);
}

cached.releaseGpu(h);                           // tombstones handle
```

- `loadGpu(buf)` returns an N-API `External` handle with a finalize callback; the underlying `DeviceBuffer` is released when the handle becomes unreachable (or sooner if you call `releaseGpu`).
- `countByteHandle(handle, targetByte)` launches the same tree-reduction kernel used by the existing one-shot `countByteGpu`, but against already-resident memory and a pre-allocated partial-sums buffer. No per-call allocation, no H2D copy, no `free()`.
- `releaseGpu(handle)` marks the handle invalid so subsequent queries throw; actual memory reclaim happens on the next GC pass or process exit.

### Results (NVIDIA H100 80GB HBM3, via RunPod)

Four paths, same buffer sizes as the other simd-search rows:

| Size | CPU SIMD (AVX-512) | GPU one-shot | **GPU cached** | Cached per-call | `loadGpu` one-time |
|------|-------------------:|-------------:|---------------:|----------------:|-------------------:|
| 1 KB   |    3.7× |   0.1× |      0.1× |  ~16 μs | 0.0 ms |
| 66 KB  |    5.9× |   1.6× |      3.2× |  ~18 μs | 0.0 ms |
| 1 MB   |   20.9× |   9.2× |     51.0× |  ~18 μs | 0.1 ms |
| 17 MB  |   42.9× |  11.3× | **527.8×** |  ~29 μs | 1.4 ms |
| 105 MB |   33.9× |   5.3× | **1030.7×** | ~94 μs | 17.2 ms |

*Speedups are relative to V8 JS on the same H100 host.*

**At 17 MB, GPU cached beats CPU SIMD by 12×. At 105 MB, GPU cached beats CPU SIMD by 30×.** This is the first time anywhere in the project that Mojo GPU has beaten Mojo CPU SIMD on a byte-scan benchmark, and it happened without changing the kernel at all — only the API shape.

### Why this flipped the result

- **105 MB in 94 μs = 1.1 TB/s effective bandwidth** — roughly 37% of H100's 3 TB/s HBM3 peak. Excellent efficiency for a byte-scan kernel with shared-memory tree reduction.
- **17 MB in 29 μs = 586 GB/s effective** — lower efficiency because kernel-launch overhead is proportionally larger at smaller sizes, but still ~14× the ~42 GB/s AVX-512 DRAM bandwidth on the same host.
- **Small buffers (1 KB, 66 KB) don't win** because per-call cost is bounded below by kernel launch (~15 μs on H100), which is comparable to what CPU SIMD finishes the whole count in.
- **The one-shot `countByteGpu` row is unchanged** — it still pays the full PCIe round trip each call. Persistent buffers are a strict superset API: users who want single-call semantics keep using `countByteGpu`; users who plan to query the same data many times use the handle API.

### When to use the handle API

The decision is a straight break-even on reused queries. The persistent path wins when:

```
N * (one_shot_cost - cached_cost)  >  loadGpu_cost
```

From the numbers above, one-shot `countByteGpu` at 105 MB is ~18 ms per call, cached is ~0.094 ms per call, so each reuse saves ~18 ms. `loadGpu` for 105 MB is 17.2 ms. Break-even at **N ≥ 1** — even a single reuse pays for the upload. At 17 MB, break-even is also **N ≥ 1** (saves ~1.3 ms per call vs 1.4 ms upload). Anything less trivial than a one-shot scan is a win.

The one case where the handle API isn't worth it is small buffers where one-shot is already fast enough — anything under ~64 KB is dominated by kernel launch overhead either way, and the API ceremony isn't worth it. Use `countByteGpu` directly for those.

### What carries over to Phase 3b

The same pattern (cached device buffer + pre-allocated result destination + N-API handle with GC finalizer) generalizes to stats and grayscale with minimal change — **Phase 3b** will port those two kernels using the same template, once the M4 performance regression in this prototype is understood (M4 Metal shows a ~2× slowdown when two addons each open their own DeviceContext; H100 CUDA does not share this constraint).

## Build & Run

```bash
pixi run bash simd-search/build.sh          # existing, one-shot API
pixi run bash simd-search/build_cached.sh   # Phase 3a persistent-buffer prototype
node simd-search/search.js                  # existing benchmark
node simd-search/search_cached.js           # 4-path comparison (JS / CPU SIMD / GPU one-shot / GPU cached)
node simd-search/test_cached.js             # regression + leak smoke test
```
