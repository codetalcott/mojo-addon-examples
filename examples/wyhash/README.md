# wyhash — Fast Non-Cryptographic Hash

Match C hash performance in ~50 lines of Mojo. The speed comes from 128-bit folded multiplies via Mojo's native `DType.uint128`.

## Functions

| Export | Description | Return Type |
|--------|-------------|-------------|
| `wyHash(buf, seed?)` | Full 64-bit hash | `BigInt` |
| `wyHash64(buf, seed?)` | Same kernel, lossy | `Number` (no BigInt overhead) |

Accepts `Buffer`, `Uint8Array`, and optional seed as `Number` or `BigInt`.

## Results (M4 Mac, CPU)

*CPU-only — `vectorize()` + `parallelize()`, no GPU.*

| Function | 1KB | 64KB | 1MB | 16MB |
|----------|-----|------|-----|------|
| **wyHash** (BigInt) | 3.7x | **52.9x** | **65.9x** | **66.2x** |
| wyHash64 (Number) | 2.9x | 45.5x | 57.8x | 58.7x |

## Build & Run

```bash
pixi run bash wyhash/build.sh
node wyhash/hash.js
```
