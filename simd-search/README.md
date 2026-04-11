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

## Build & Run

```bash
pixi run bash simd-search/build.sh
node simd-search/search.js
```
