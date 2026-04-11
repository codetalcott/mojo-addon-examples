# Mojo Addon Examples

High-performance Node.js addon examples built with [napi-mojo](https://github.com/codetalcott/napi-mojo). Each example demonstrates Mojo's SIMD `vectorize()` and `parallelize()` through the N-API bridge, with benchmarks against pure JavaScript.

## Examples

> **Benchmarking scope:** The primary CPU numbers in each table below are measured on an Apple M4. GPU rows report the M4's integrated GPU via Metal 4. A separate [H100 Cloud Benchmark Results](#h100-cloud-benchmark-results) section below reports results from an NVIDIA H100 80GB HBM3 (rented via RunPod). **Headline finding from the cloud run: Mojo CPU SIMD beats Mojo GPU on every benchmark — on both M4 Metal *and* H100 CUDA.** These addons call the GPU as one-shot N-API functions against host-resident data, which is PCIe-bound rather than HBM-bound. See [per-addon READMEs](#statistics--simd-aggregation) and the H100 section below for the full teardown.

### Matrix Multiply — Progressive Optimization

Four implementations showing Mojo's optimization story, from naive triple loop to SIMD + tiled + parallel:

```
node matmul/matmul.js
```

**Results (M4 Mac, CPU, Float64):**

| Step | 1024x1024 | 2048x2048 | Mojo Feature |
|------|-----------|-----------|-------------|
| JS baseline | 1x | 1x | — |
| Mojo naive | 2.2x | 1.9x | Same algorithm, better compiler |
| Mojo vectorized | 15.5x | 32.0x | `vectorize()` — SIMD inner loop |
| Mojo tiled | 12.6x | 27.6x | Cache-friendly 64x64 blocking |
| **Mojo parallel** | **38.6x** | **91.4x** | `parallelize()` — multi-core |

### SIMD Text Search — Byte-Level Pattern Matching

SIMD byte scanning that's impossible to express in pure JavaScript. Three functions: `countByte`, `countLines`, `searchAll` (single and multi-byte patterns).

```
node simd-search/search.js
```

**Results (M4 Mac, Buffer/Uint8Array):**

| Function | 1MB | 16MB | 100MB | Mojo Feature |
|----------|-----|------|-------|-------------|
| **countByte CPU** | **19.2x** | **52.4x** | **67.6x** | SIMD XOR + reduce, `parallelize()` |
| countByte GPU Metal | 0.9x | 1.0x | 2.3x | Tree reduction, shared memory partials |
| countLines | 18.5x | 50.1x | 65.3x | Same kernel, byte=0x0A |
| searchAll (1-byte) | 2.5x | 2.8x | 3.1x | Two-pass: SIMD count + collect |
| searchAll (multi-byte) | 2.0x | 2.3x | 2.5x | First+last byte SIMD filter |

`countByteGpu()` is published honestly: at 100MB it's 2.3× JS vs CPU SIMD's 67.6× — a 30× gap. Byte-scan kernels on integrated GPUs are copy-bound rather than compute-bound; the M4 CPU's direct DRAM access wins decisively. Output matches the CPU path exactly. See [simd-search/README.md](simd-search/README.md) for why, and expected-to-flip-on-H100 notes.

### Statistics — SIMD Aggregation

Compute `{mean, stddev, min, max, p50, p95, p99}` on Float64Arrays in a single call. SIMD reductions + parallel accumulation + quickselect percentiles.

```
node stats/stats.js
```

**Results (M4 Mac, Float64):**

| Function | 100K | 1M | 10M | Mojo Feature |
|----------|------|-----|-----|-------------|
| **stats() CPU** | **4.2x** | **5.8x** | **6.7x** | SIMD reduce_add/min/max, `parallelize()` |
| stats() GPU Metal | 2.1x | 4.0x | 4.2x | Shared-memory tree reduction via `DeviceContext` (Float32 internal) |
| histogram() CPU | 3.7x | 3.9x | 4.0x | SIMD min/max range detection |

`statsGpu()` runs on the M4's integrated GPU via Mojo's Metal 4 backend. Kernels are Float32 (Metal constraint); the H2D cast and final Float64 reduction happen on the host. See [stats/README.md](stats/README.md) for precision notes.

### Image Processing — Pixel Operations

Four RGBA pixel operations on Uint8Arrays: `grayscale`, `brightness`, `threshold`, `blur`. Integer-approximation grayscale, fixed-point brightness, separable box blur with parallel horizontal + vertical passes.

```
node image/image.js
```

**Results (M4 Mac, RGBA Uint8Array):**

| Function | 720p | 1080p | 4K | Mojo Feature |
|----------|------|-------|-----|-------------|
| grayscale CPU | 6.8x | 5.4x | **6.8x** | Integer `(77R+150G+29B)>>8`, `parallelize()` |
| grayscale GPU Metal | 0.6x | 0.6x | 0.8x | One-pixel-per-thread elementwise kernel |
| brightness | 4.7x | 5.1x | 5.0x | Fixed-point multiply + clamp, `parallelize()` |
| threshold | 5.5x | 5.2x | **6.5x** | Grayscale + compare, `parallelize()` |
| **blur(r=5)** | 6.8x | **10.1x** | 5.6x | Separable box blur, parallel rows + cols |

`grayscaleGpu()` is published despite being **slower than JS** because it illustrates an important limit: on integrated GPUs (M4's unified memory architecture), the H2D/D2H "copies" are pure overhead with no bandwidth benefit. For a trivial per-pixel kernel the copy cost dominates. Output matches the CPU path byte-for-byte. On a discrete GPU with real HBM the ordering is expected to flip — see [image/README.md](image/README.md).

### wyhash — Fast Non-Cryptographic Hash

Match C hash performance in ~50 lines of Mojo. `wyHash` returns BigInt (full 64-bit), `wyHash64` returns Number (lossy but no BigInt allocation overhead). The speed comes from 128-bit folded multiplies via Mojo's native `DType.uint128`.

```
node wyhash/hash.js
```

**Results (M4 Mac, CPU, Buffer):**

| Function | 1KB | 64KB | 1MB | 16MB | Mojo Feature |
|----------|-----|------|-----|------|-------------|
| **wyHash** (BigInt) | 3.7x | **52.9x** | **65.9x** | **66.2x** | 128-bit folded multiply |
| wyHash64 (Number) | 2.9x | 45.5x | 57.8x | 58.7x | Same kernel, Number return |

## H100 Cloud Benchmark Results

We ran the three GPU-enabled addons (stats, image, simd-search) on an **NVIDIA H100 80GB HBM3** via RunPod (~$1.99/hr) to see whether the M4 Metal story — GPU losing to CPU SIMD — would flip on Tier-1 discrete GPU hardware. **It did not.** On every benchmark, Mojo CPU SIMD running on the H100 host's Xeon (AVX-512) beat Mojo GPU running on the H100.

The reason matters more than the numbers. These addons call the GPU as one-shot N-API functions against **host-resident** data. Every call does `alloc → H2D copy → kernel → D2H copy → free`. On a PCIe Gen4 x16 link (~12 GB/s effective), PCIe is the bottleneck for data transfer, and the H100's 3 TB/s HBM3 bandwidth is effectively wasted because the data never lives on-device long enough to amortize it. Meanwhile the Xeon host's AVX-512 reads the same data at ~30 GB/s directly from DRAM. For low-arithmetic-intensity kernels (reductions, elementwise, byte scanning) on a single-call API, CPU wins.

This is the opposite of the narrative in most "port to GPU" benchmarks, and it's the most useful finding from the cloud run.

### Stats on H100 (Float64 arrays, ratios vs V8 on the same host)

| Size | H100 Mojo SIMD | H100 Mojo GPU |
|------|----------------|---------------|
| 100K | 5.0× | 4.7× |
| 1M | 5.3× | 5.2× |
| 10M | **8.3×** | 7.6× |

Two factors drag stats on GPU: (1) the Float64→Float32 cast runs as a scalar host loop at [stats/addon.mojo:327](stats/addon.mojo#L327) — at 10M elements this alone is tens of ms per call — and (2) stats does two separate passes (`{sum,min,max}` then `sum_sq_diff(mean)`), each with its own allocation + copy cycle, doubling per-call overhead.

### Image grayscale on H100 (4K RGBA)

| Size | H100 Mojo SIMD | H100 Mojo GPU |
|------|----------------|---------------|
| 720p | 3.2× | 2.6× |
| 1080p | 4.0× | 2.1× |
| 4K | **4.3×** | 1.4× |

The GPU gets *worse* relative to CPU as size grows. At 720p both paths bottleneck on per-call overhead so they're close; at 4K, CPU SIMD scales with DRAM bandwidth (~30 GB/s effective via AVX-512) while the GPU is PCIe-capped on 33 MB H2D + 33 MB D2H each call. Grayscale is the canonical memory-bandwidth-bound elementwise kernel where PCIe decides everything.

### countByte on H100 (host Buffer scan)

| Size | H100 Mojo SIMD | H100 Mojo GPU |
|------|----------------|---------------|
| 1MB | **60.1×** | 9.2× |
| 17MB | **89.6×** | 12.2× |
| 105MB | **35.1×** | 6.0× |

AVX-512 byte scanning on Xeon Sapphire Rapids is brutal for this workload — 89.6× at 17MB is the most dramatic CPU-SIMD speedup anywhere in the project, and the H100 GPU trails by 7×. The 1KB row (GPU 0.0×, 25.7 μs per call) is pure CUDA API overhead: cached `DeviceContext` can't get below the kernel launch + device allocation cost, and CPU SIMD completes the same count in 357 ns.

### What would actually make H100 win

1. **Persistent device-resident buffers.** Upload a dataset once, scan it a thousand times. PCIe amortizes away and HBM bandwidth dominates.
2. **Batched N-API API.** Process N inputs per call with async copy-compute overlap so the GPU is never idle waiting for PCIe.
3. **Higher arithmetic intensity.** Matmul (O(n³) compute on O(n²) data), convolution, attention — anything where the compute cost dominates the transfer cost. Stats, grayscale, and countByte are all ≤1 op per byte, the worst possible ratio for any GPU on a PCIe link.

All three are Phase 3 candidate work. See [docs/cloud-benchmark-runbook.md](docs/cloud-benchmark-runbook.md) to reproduce these numbers (~$1, ~30 minutes on RunPod), and the per-addon READMEs for deeper teardowns.

## When to Use Mojo

V8's JIT compiler is already fast for scalar code. The matmul example shows this clearly: Mojo with the *same algorithm* is only 1.9-2.2x faster. A native addon has real costs -- build toolchain, N-API call overhead, platform-specific binaries. Mojo is worth reaching for when:

**The data is large and the work is data-parallel.** Speedups scale with input size across every example: wyhash is 3.7x at 1KB but 66x at 16MB. countByte is 19x at 1MB but 68x at 100MB. If your hot loop processes a TypedArray or Buffer with thousands of elements, Mojo's SIMD `vectorize()` can process 2-8 elements per instruction where V8 processes one.

**You need multi-core parallelism.** V8 is single-threaded. Worker threads exist but require serialization overhead. Mojo's `parallelize()` distributes work across cores with zero-copy shared memory. The matmul example jumps from 15x (SIMD only) to 91x (SIMD + parallel) by adding one line.

**The operation can't be expressed in JS.** Byte-level SIMD (XOR + reduce for pattern matching), 128-bit integer arithmetic (wyhash's folded multiply), and fixed-point pixel math all require bit-width control that JavaScript doesn't offer. These aren't just faster -- they're impossible to write in JS at all.

**When NOT to use Mojo:** String manipulation, JSON parsing, I/O-bound work, small payloads where N-API call overhead dominates, or anything V8 already JIT-compiles well. If your function runs in under ~1ms on typical input, the native call overhead likely isn't worth it.

## Prerequisites

- [pixi](https://prefix.dev/docs/pixi/) with Mojo nightly
- Node.js 18+
- **For GPU benchmarks on Apple Silicon:** Xcode with the Metal Toolchain component installed (`sudo xcode-select -s /Applications/Xcode.app/Contents/Developer && xcodebuild -downloadComponent MetalToolchain`). Builds default to `--target-accelerator metal:4` on Darwin arm64; override per-addon with `STATS_ACCEL=""` etc. to build CPU-only.
- **For GPU benchmarks on Linux/NVIDIA:** builds default to `--target-accelerator sm_90` (H100/H200). For other NVIDIA architectures, set `STATS_ACCEL="--target-accelerator sm_80"` (A100), `sm_89` (L40/RTX40), etc.
- **Rented cloud GPU walkthrough:** see [docs/cloud-benchmark-runbook.md](docs/cloud-benchmark-runbook.md) for a ~30-minute Lambda Cloud H100 benchmark flow.

## Quick Start

```bash
npm install
pixi install

# Build all examples
npm run build:all

# Run benchmarks
node matmul/matmul.js
node simd-search/search.js
node stats/stats.js
node image/image.js
node wyhash/hash.js
```

## Development

To test examples against a local (unreleased) version of napi-mojo:

```bash
cd /path/to/napi-mojo && npm link
cd /path/to/mojo-addon-examples && npm link napi-mojo
```

This replaces the npm-installed package with a symlink to your local checkout. Run `npm install` to revert back to the published package.

## Architecture

Each example is self-contained:

```text
example-name/
  addon.mojo          # Mojo source (SIMD kernels + N-API callbacks)
  example.js           # Demo + benchmark script
  build.sh             # Build script (compile .mojo -> .node)
  README.md            # Example-specific docs + benchmark results
```

All examples depend on napi-mojo for the N-API framework (`napi.types`, `napi.framework.*`).

### Common Patterns

- **Zero-copy TypedArray access:** `JsTypedArray.data_ptr(env).bitcast[Float64]()` reads JS memory directly
- **SIMD vectorize:** `vectorize[simd_width_of[DType.float64]()](size, compute)` with `unified {mut}` closure
- **Multi-core parallel:** `parallelize[worker](num_workers)` with `capturing` closure
- **Runtime init:** `KGEN_CompilerRT_AsyncRT_CreateRuntime` via `OwnedDLHandle()` for parallelize in shared libs
