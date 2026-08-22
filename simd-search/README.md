# SIMD Text Search — Byte-Level Pattern Matching

SIMD byte scanning that's impossible to express in pure JavaScript. XOR-based byte matching with `parallelize()` for large buffers.

## Functions

| Export | Description | Mojo Feature |
|--------|-------------|-------------|
| `countByte(buf, byte)` | Count occurrences of a byte | SIMD XOR + reduce, `parallelize()` |
| `countLines(buf)` | Count newlines (`wc -l`) | Same kernel, byte=0x0A |
| `searchAll(buf, needle)` | All match positions as `Uint32Array` | Two-pass: SIMD count + collect |

Accepts both `Buffer` and `Uint8Array` inputs. Multi-byte `searchAll` uses a first+last byte SIMD filter to minimize full comparisons.

## Results (M4 Mac)

| Function | 1MB | 16MB | 100MB |
|----------|-----|------|-------|
| **countByte** | **19.2x** | **52.4x** | **67.6x** |
| countLines | 18.5x | 50.1x | 65.3x |
| searchAll (1-byte) | 2.5x | 2.8x | 3.1x |
| searchAll (multi-byte) | 2.0x | 2.3x | 2.5x |

## Build & Run

```bash
npm run build:search
node simd-search/search.js
```
