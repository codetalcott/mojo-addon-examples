# Phase 3c.1 — Mojo tensor-core matmul feasibility research

**Date**: 2026-04-11
**Scope**: single-day research sprint to answer: *can we ship a FP16 tensor-core matmul in Mojo within ~1 week of Phase 3c.2 work, or does the stdlib require >2 weeks of MLIR hand-rolling?*

## TL;DR — **GREEN. Proceed to 3c.2.**

Mojo's `layout.tensor_core.TensorCore` struct exists in our pixi env today and exposes a high-level wmma wrapper with `load_a` / `load_b` / `load_c` / `mma_op` / `store_d` methods taking plain `LayoutTensor` arguments. An **official Modular tutorial** (Mojo GPU Puzzles #33) shows a complete tensor-core matmul kernel using this exact API — the inner-loop body is ~8 lines. A **second official tutorial** (MAX custom-ops-matmul) documents the same pattern end-to-end. No MLIR hand-rolling required. Kill criterion not tripped.

Recommended 3c.2 target: **FP16 inputs + FP32 accumulation**, MMA shape **16×8×16** (NVIDIA half-precision), starting with a single-block 128×128×128 prototype, scaling up to 1024/2048/4096 with standard shared-memory tiling. Stretch goal: Hopper-specific `TensorCoreAsync` (WGMMA) + `TMATensorTile` (async bulk copies) if 3c.2 comes in under budget.

## What's in our pixi env (confirmed)

```text
.pixi/envs/default/lib/mojo/
├── std.mojopkg
├── layout.mojopkg      ← tensor core primitives live here
└── buffer.mojopkg
```

`layout.mojopkg` contains (verified via `mojo doc` extraction):

| Module | What it gives us |
|---|---|
| `layout.tensor_core` | `TensorCore`, `TiledTensorCore` — wmma fragment wrappers (NVIDIA + AMD) |
| `layout.tensor_core_async` | `TensorCoreAsync` — Hopper SM90 WGMMA (async mma) |
| `layout.tma_async` | `TMATensorTile`, `SharedMemBarrier`, `PipelineState` — TMA bulk async copies + producer-consumer sync |
| `layout.swizzle` | Shared memory swizzle patterns for bank conflict avoidance |
| `layout.layout_tensor` | `LayoutTensor` (the primary GPU data abstraction — already used in Phase 3a/3b) |

## `TensorCore` API surface

Full signature (from mojo stdlib docs):

```mojo
struct TensorCore[
    out_type: DType,         # accumulator type (typically f32)
    in_type: DType,           # input type (f16, bf16, f32, f8)
    shape: IndexList[3],      # [M, N, K] tile dimensions
    transpose_b: Bool = False
]:
    fn __init__(out self): ...

    # Load matrix fragments — A, B from shared/global memory, C from accumulator
    fn load_a(self, a: LayoutTensor[...]) -> a_reg_tile_type
    fn load_b(self, b: LayoutTensor[...]) -> b_reg_tile_type
    fn load_c(self, c: LayoutTensor[...]) -> c_reg_tile_type

    # D = A @ B + C, returning a new accumulator fragment
    fn mma_op(self, a, b, c) -> c_reg_tile_type

    # In-place variant — accumulates into c_frag
    fn mma(self, a_frag, b_frag, c_frag)

    # Write accumulator back to memory
    fn store_d(self, d_dst: LayoutTensor[...], d_src: LayoutTensor[...])
```

**Supported shapes** (per `TensorCore` docs note):
- NVIDIA FP32: `shape_16x8x8` or `shape_16x8x4`
- NVIDIA **FP16 / BF16**: `shape_16x8x16` ← **3c.2 target**
- NVIDIA FP8: `shape_16x8x32`
- AMD FP32: `shape_16x16x4`
- AMD FP16: `shape_16x16x16` or `shape_32x32x8`

Shape aliases exist at module scope: `shape_16x8x4`, `shape_16x8x8`, `shape_16x8x16`, `shape_16x8x32`, `shape_8x8x4`, `shape_16x16x4`, `shape_16x16x16`, `shape_16x16x32`, `shape_32x32x8`, `shape_32x32x16`.

## Reference examples (both official Modular resources)

### 1. Mojo GPU Puzzles #33 — Tensor Core matmul

URL: <https://puzzles.modular.com/puzzle_33/puzzle_33.html>

Transforms a tiled matmul into a tensor-core implementation. Uses **MMA shape 16×8×8 for FP32** (not FP16), with block tiling **BM=128, BN=64, BK=32** and warp tiling **WM=32, WN=32** on a **1024×1024** matrix. The kernel body's inner MMA step is literally this:

```mojo
var mma_op = TensorCore[A.dtype, C.dtype, Index(MMA_M, MMA_N, MMA_K)]()

# inside the K-tile loop, per warp-tile (mma_m, mma_n, mma_k):
var A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
var B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](mma_k, mma_n)
var C_mma_tile = C_warp_accum.tile[MMA_M, MMA_N](mma_m, mma_n)

var a_reg = mma_op.load_a(A_mma_tile)
var b_reg = mma_op.load_b(B_mma_tile)
var c_reg = mma_op.load_c(C_mma_tile)
var d_reg = mma_op.mma_op(a_reg, b_reg, c_reg)
mma_op.store_d(C_mma_tile, d_reg)
```

This is the exact API pattern we'll adapt for FP16 in 3c.2. The difference: `A.dtype = DType.float16`, `C.dtype = DType.float32`, `Index(16, 8, 16)` instead of `Index(16, 8, 8)`.

### 2. MAX custom-ops-matmul tutorial

URL: <https://docs.modular.com/max/develop/custom-ops-matmul/>

Progressive optimization walkthrough ending in a tensor-core kernel using the same `TensorCore` API. Reports **~2× speedup on A100** over the vectorized block-tiled version (not over JS — over Mojo's own block-tiled baseline). Shows the full host-side setup including `ctx.enqueue_function[...]`, tile dimension choices, and thread block configuration.

### 3. Other references worth pointing at

- <https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction> — Modular blog on SM100 (Blackwell) matmul, useful if we want to understand the TMA + WGMMA path for the async variant
- LLVM 2024 dev meeting slides: <https://llvm.org/devmtg/2024-10/slides/techtalk/Taei-Simplifying-GPU-Programming-with-Parametric-Tile-Level-Tensors-In-Mojo.pdf> — Mojo team's own talk on parametric tile tensors, gives the design rationale

## What is NOT in our pixi env

The Mojo docs also document a much higher-level matmul kernel library: `gemm_kernel_amd`, `batched_matmul_kernel_gpu`, `naive_block_scaled_matmul_kernel`, `grouped_matmul_kernel_sm100`, `MatmulConfig`, `DefaultMatmulOps`, etc. These live under `kernels.linalg.matmul` in the docs.

**They are not shipped as a separate `.mojopkg` in our env** — only `std`, `layout`, and `buffer`. This means we cannot `import` MAX's production matmul and call it directly. Instead, we build our own kernel from `layout.tensor_core.TensorCore` primitives. Fortunately the Puzzle 33 + custom-ops-matmul tutorials show exactly how to do that.

If we later want to use the higher-level MAX kernel library, we'd need to integrate with MAX's custom-op system (which has its own N-API surface and doesn't fit the napi-mojo pattern we're using). That's Phase 4+ work — out of scope for 3c.

## Proposed 3c.2 plan

Split into three sub-milestones with clear gates:

### 3c.2a — Single-block prototype (2 days)

- New file: `matmul/addon_cached.mojo` (does NOT share with existing CPU `matmul/addon.mojo`)
- Start with **M=N=K=128**, single thread block (`grid_dim=(1,1)`, `block_dim=128` or similar)
- FP16 inputs cast from JS Float32Array, FP32 accumulation, FP16 output cast back
- Kernel body copied structurally from Puzzle 33 but with `Index(16, 8, 16)` for FP16
- Correctness target: byte-identical to a CPU Float32 reference within 1e-3 relative tolerance on uniform random inputs
- **Gate**: kernel compiles, runs without crash, correctness within tolerance. If this doesn't work, debug for 1 more day then abandon 3c per strategy doc kill criterion.

### 3c.2b — Scale to 1024 × 1024 × 1024 with block tiling (2 days)

- Add block-level tiling: **BM=128, BN=128, BK=32** (standard cuBLAS-style pattern)
- Each thread block has 4 warps (128 threads); each warp handles a 64×64 output tile using 4×8 MMAs of shape 16×8×16
- Shared memory double buffering: load next K tile while computing current
- Correctness re-verified at 1024 against FP32 CPU path within 1e-3 rel tol
- **Gate**: correctness + no crash at 1024. Performance is secondary — measure but don't gate on it.

### 3c.2c — Persistent buffer wrapper (1 day)

Layer the Phase 3a persistent-buffer template on top:

```js
const hA = cached.loadMatrixGpu(aF32, M, K);
const hB = cached.loadMatrixGpu(bF32, K, N);
const cBuf = new Float32Array(M * N);
for (let i = 0; i < 100; i++) {
  cached.matmulHandle(hA, hB, cBuf);    // reuses A, B, allocates no device memory per call
}
cached.releaseMatrixGpu(hA);
cached.releaseMatrixGpu(hB);
```

- New `CachedMatrix` struct holding `DeviceBuffer[DType.float16]` (cast + uploaded once at load time)
- `matmulHandle(hA, hB, cBuf)` entry: validates both externals, checks shape compatibility, runs kernel, D2H copy to caller-provided Float32 buffer
- Handle struct owns the FP16 buffer; finalize runs `destroy_pointee` + `free` as usual
- Copied plumbing from 3b.1/3b.2 templates (~150 lines)

**Gate**: correctness at 1024 × 1024 via `matmulHandle` path, leak-smoke doesn't crash.

### 3c.3 — H100 benchmark (1 day, user-driven)

Same workflow as 3b.3 Track B:
- Extend `scripts/runpod-bench-3b.sh` with matmul cached build + bench OR write a dedicated `scripts/runpod-bench-3c.sh`
- Sizes: **1024², 2048², 4096²**
- Paths: JS, Mojo CPU parallel (existing `matmul/addon.mojo`), Mojo GPU cached FP16 tensor core
- Expected on H100 FP16 tensor core peak is 989 TFLOPS. At 50% realistic efficiency = 500 TFLOPS. 4096³ = 137 GFLOPs → **~0.27 ms** pure compute. Adding PCIe D2H for the 64 MB FP32 C matrix at 12 GB/s = ~5.3 ms. Realistic per-call wall clock: **~5-10 ms at 4096²**.
- JS pure matmul at 4096² is ~8-30 s (V8 single-threaded, no SIMD).
- Predicted speedup: **800×–6000× JS at 4096²**, solidly above the strategy doc's Green gate (≥100× JS, ≥5× CPU SIMD).

## Risks

1. **FP16 precision loss**: ~1e-3 relative error vs FP32 CPU on uniform random inputs. This is the standard FP16 accuracy floor for matmul. Document prominently in README — the "Mojo GPU matmul is 800× JS" claim only holds at FP16. Users needing FP32-strict results use the existing CPU `matmul.node` path.

2. **Puzzle 33 uses FP32, not FP16**: the reference example validates the API shape but the specific shape constant is different (16×8×8 vs 16×8×16). Risk: FP16 path may hit a case the tutorial doesn't cover (e.g., fragment layouts differ between FP32 and FP16 in ways the tutorial doesn't surface). Mitigation: start with a single-block prototype so any FP16-specific issues surface early.

3. **sm_90 architecture flag**: H100 builds use `--target-accelerator sm_90`. The TensorCore primitive should emit base wmma intrinsics compatible with any SM70+ target, not Hopper-specific ones. But WGMMA (if we stretch to it) is sm_90-only. Worth confirming with a compile test on the A100 docs path (`sm_80`) even though we won't run there.

4. **LayoutTensor fragment register type mismatch**: the `TensorCore` methods return `c_reg_tile_type` which is a `LayoutTensor` with a computed layout. Feeding this back into `mma_op` as the `c` argument of the next K-tile iteration requires the layouts to line up. Puzzle 33's pattern (accumulating in `C_warp_accum` and then per-tile `load_c`/`store_d`) handles this but may produce subtle bugs if we tile it wrong.

5. **M4 Metal support is unknown**. Mojo's Metal backend is Tier-3 and may not expose tensor cores (Apple's matrix coprocessor lives in AMX, accessed via `simdgroup_matrix` in MSL — unclear if Mojo's Metal backend passes `TensorCore` calls through to it). **Assume 3c.2 is H100-only**. The build should gate via `comptime if is_nvidia_gpu()` or similar; on M4, the cached addon falls back to a CPU error. This mirrors the strategy doc's explicit assumption.

6. **Leak behavior unknown for FP16 buffers**. 3b.1 and 3b.2 both leak ~1 DeviceBuffer per load+release cycle (Track A finding). Matmul adds 2 DeviceBuffers (A and B handles) so the leak may double. Not blocking — production usage is load-once-query-many, but worth noting.

## Time budget

| Milestone | Days | Gate |
|---|---|---|
| 3c.2a — single-block prototype | 2 | Correctness on 128³ |
| 3c.2b — scale to 1024³ w/ block tiling | 2 | Correctness on 1024³ |
| 3c.2c — persistent handle wrapper + test | 1 | Correctness via `matmulHandle` |
| 3c.3 — H100 bench + README updates | 1 | Green gate: ≥100× JS at 4096² |
| **Total** | **6 days** | |

Comfortably under the strategy doc's "~1 week" target. Kill criterion (>2 weeks MLIR infra) not tripped.

## Decision: GREEN — proceed to Phase 3c.2

All four strategy-doc research questions answered:

1. **Does std.gpu expose wmma-style primitives?** Not directly under `std.gpu`, but `layout.tensor_core.TensorCore` in the `layout.mojopkg` that ships with our Mojo nightly provides a user-level wrapper with `load_a`/`load_b`/`mma_op`/`store_d`. ✓

2. **Highest-level API?** `TensorCore` is the right level for a first kernel. `TensorCoreAsync` (WGMMA) is Hopper-specific and more complex — reserve as stretch goal. The MAX kernel library's `gemm_kernel_amd` / `matmul(Matmul)` is higher-level still but lives outside our pixi env. ✓

3. **Metal comparison?** Assumed H100-only for 3c. M4 Metal path falls back or errors. ✓

4. **LayoutTensor + persistent buffer interaction?** LayoutTensor wraps a DeviceBuffer directly via `LayoutTensor[dtype, layout](buf)`. Persistent buffer stores the FP16 DeviceBuffer; per-call kernel constructs LayoutTensor views on it. No tile rearrangement at upload time needed — the kernel handles tile extraction per call via `.tile[M, K](row, col)`. ✓

No blockers. Proceeding to 3c.2 is well-scoped 6-day work.

## Appendix: what 3c.2a's first file might look like

Rough sketch (not committed, just for sizing — actual file will be in 3c.2):

```mojo
from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import thread_idx, block_idx, barrier
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.utils.index import Index
from layout import Layout, LayoutTensor
from layout.tensor_core import TensorCore

# FP16 + FP32 accumulation, NVIDIA shape
comptime in_type = DType.float16
comptime out_type = DType.float32
comptime MMA_M = 16
comptime MMA_N = 8
comptime MMA_K = 16
comptime mma_shape = Index(MMA_M, MMA_N, MMA_K)

# Block tiling (single-block prototype)
comptime BM = 128
comptime BN = 128
comptime BK = 32
comptime WM = 64    # warp tile M
comptime WN = 64    # warp tile N
comptime NUM_WARPS = 4
comptime BLOCK_THREADS = NUM_WARPS * 32   # 128 threads / block

comptime a_layout = Layout.row_major(BM, BK)
comptime b_layout = Layout.row_major(BK, BN)
comptime c_layout = Layout.row_major(BM, BN)

def matmul_kernel_fp16[M: Int, N: Int, K: Int](
    A: LayoutTensor[in_type, Layout.row_major(M, K), MutAnyOrigin],
    B: LayoutTensor[in_type, Layout.row_major(K, N), MutAnyOrigin],
    C: LayoutTensor[out_type, Layout.row_major(M, N), MutAnyOrigin],
):
    var mma_op = TensorCore[out_type, in_type, mma_shape]()

    # Allocate A and B tiles in shared memory
    var sA = LayoutTensor[in_type, a_layout, MutAnyOrigin,
        address_space=AddressSpace.SHARED].stack_allocation()
    var sB = LayoutTensor[in_type, b_layout, MutAnyOrigin,
        address_space=AddressSpace.SHARED].stack_allocation()

    # Per-warp accumulator (in registers)
    # ... warp tile extraction ...
    # ... K-tile loop with Puzzle 33 style inner MMA body ...
    # ... final store back to C ...
```

This is ~100 lines total for 3c.2a when filled in. Phase 3c.2b adds another ~100 lines (proper block + warp tiling across multiple blocks). Phase 3c.2c adds ~150 lines of N-API cached-handle plumbing copied from 3b.1/3b.2. Total ~350–400 lines for the full Phase 3c.2 addon.
