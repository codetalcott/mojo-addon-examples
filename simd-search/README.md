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

**GPU countByte is dramatically slower than CPU SIMD on M4** — at 100MB it's 2.3× JS vs CPU SIMD's 67.6×, a **30× gap**. This is the worst-case shape for integrated-GPU computation: the kernel is trivial (one compare + increment per byte), the data is huge (H2D copy dominates), and the CPU side already has fast SIMD + multi-core parallelism over direct DRAM. The GPU result is correct — output matches CPU byte-count exactly — but the M4 doesn't have the bandwidth headroom to amortize the dispatch cost on a byte-scan kernel. Expected to flip on H100 (3TB/s HBM) where GPU becomes compute-bound on the reduction rather than copy-bound.

## Build & Run

```bash
pixi run bash simd-search/build.sh
node simd-search/search.js
```
