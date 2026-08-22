# Mojo Addon Examples

High-performance Node.js addon examples built with [napi-mojo](https://github.com/codetalcott/napi-mojo). Each example demonstrates Mojo's SIMD `vectorize()` and `parallelize()` through the N-API bridge, with benchmarks against pure JavaScript.

## Examples

All figures are speedups over pure JavaScript, measured on an M4 Mac.

### Matrix Multiply — Progressive Optimization

Four implementations showing Mojo's optimization story, from naive triple loop to SIMD + tiled + parallel. Run with `npm run demo:matmul`.

| Step | 1024x1024 | 2048x2048 | Mojo Feature |
|------|-----------|-----------|-------------|
| JS baseline | 1x | 1x | — |
| Mojo naive | 2.2x | 1.9x | Same algorithm, better compiler |
| Mojo vectorized | 15.5x | 32.0x | `vectorize()` — SIMD inner loop |
| Mojo tiled | 12.6x | 27.6x | Cache-friendly 64x64 blocking |
| **Mojo parallel** | **38.6x** | **91.4x** | `parallelize()` — multi-core |

### SIMD Text Search — Byte-Level Pattern Matching

SIMD byte scanning that's impossible to express in pure JavaScript. Three functions: `countByte`, `countLines`, `searchAll` (single and multi-byte patterns). Run with `npm run demo:search`.

| Function | 1MB | 16MB | 100MB | Mojo Feature |
|----------|-----|------|-------|-------------|
| **countByte** | **19.2x** | **52.4x** | **67.6x** | SIMD XOR + reduce, `parallelize()` |
| countLines | 18.5x | 50.1x | 65.3x | Same kernel, byte=0x0A |
| searchAll (1-byte) | 2.5x | 2.8x | 3.1x | Two-pass: SIMD count + collect |
| searchAll (multi-byte) | 2.0x | 2.3x | 2.5x | First+last byte SIMD filter |

### Statistics — SIMD Aggregation

Compute `{mean, stddev, min, max, p50, p95, p99}` on Float64Arrays in a single call. SIMD reductions + parallel accumulation + quickselect percentiles. Run with `npm run demo:stats`.

| Function | 100K | 1M | 10M | Mojo Feature |
|----------|------|-----|-----|-------------|
| **stats()** | **4.2x** | **5.8x** | **6.7x** | SIMD reduce_add/min/max, `parallelize()` |
| histogram() | 3.7x | 3.9x | 4.0x | SIMD min/max range detection |

### Image Processing — Pixel Operations

Four RGBA pixel operations on Uint8Arrays: `grayscale`, `brightness`, `threshold`, `blur`. Integer-approximation grayscale, fixed-point brightness, separable box blur with parallel horizontal + vertical passes. Run with `npm run demo:image`.

| Function | 720p | 1080p | 4K | Mojo Feature |
|----------|------|-------|-----|-------------|
| grayscale | 6.8x | 5.4x | **6.8x** | Integer `(77R+150G+29B)>>8`, `parallelize()` |
| brightness | 4.7x | 5.1x | 5.0x | Fixed-point multiply + clamp, `parallelize()` |
| threshold | 5.5x | 5.2x | **6.5x** | Grayscale + compare, `parallelize()` |
| **blur(r=5)** | 6.8x | **10.1x** | 5.6x | Separable box blur, parallel rows + cols |

### wyhash — Fast Non-Cryptographic Hash

Match C hash performance in ~50 lines of Mojo. `wyHash` returns BigInt (full 64-bit), `wyHash64` returns Number (lossy but no BigInt allocation overhead). The speed comes from 128-bit folded multiplies via Mojo's native `DType.uint128`. Run with `npm run demo:hash`.

| Function | 1KB | 64KB | 1MB | 16MB | Mojo Feature |
|----------|-----|------|-----|------|-------------|
| **wyHash** (BigInt) | 3.7x | **52.9x** | **65.9x** | **66.2x** | 128-bit folded multiply |
| wyHash64 (Number) | 2.9x | 45.5x | 57.8x | 58.7x | Same kernel, Number return |

See [ROADMAP.md](ROADMAP.md) for the full project roadmap.

## When to Use Mojo

V8's JIT is already fast for scalar code — Mojo running the *same algorithm* is only 1.9-2.2x faster, and a native addon costs you a build toolchain, N-API call overhead, and platform-specific binaries. Mojo is worth reaching for when:

**The data is large and the work is data-parallel.** Speedups scale with input size across every example above. Mojo's SIMD `vectorize()` processes 2-8 elements per instruction where V8 processes one.

**You need multi-core parallelism.** V8 is single-threaded, and worker threads require serialization overhead. Mojo's `parallelize()` distributes work across cores with zero-copy shared memory — matmul goes from 15x to 91x by adding one line.

**The operation can't be expressed in JS.** Byte-level SIMD, 128-bit integer arithmetic, and fixed-point pixel math all need bit-width control JavaScript doesn't offer. These aren't just faster — they're impossible to write in JS at all.

**When NOT to use Mojo:** String manipulation, JSON parsing, I/O-bound work, small payloads where N-API call overhead dominates, or anything V8 already JIT-compiles well. If your function runs in under ~1ms on typical input, the native call overhead likely isn't worth it.

## Quick Start

Requires [pixi](https://prefix.dev/docs/pixi/), which provides the pinned Mojo toolchain, and Node.js 22.12+.

```bash
npm install
pixi install
npm run build:all
npm test
```

Then run any of the demos listed above.

## Development

To test examples against a local (unreleased) version of napi-mojo:

```bash
cd /path/to/napi-mojo && npm link
cd /path/to/mojo-addon-examples && npm link napi-mojo
```

Run `npm install` to revert to the published package.
