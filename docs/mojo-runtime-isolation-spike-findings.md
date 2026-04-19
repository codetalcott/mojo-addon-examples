# Mojo runtime isolation spike — findings

**Branch:** `spike/mojo-runtime-isolation`
**Date:** 2026-04-17
**Toolchain:** Mojo `0.26.3.0.dev2026041716`, M4 Metal (Darwin arm64)
**Goal:** answer a question raised while drafting outreach to Modular's licensing team — *if Modular declines our request to bundle MAX runtime libs in `@qkstat/rag`, can we eliminate the MAX dependency by stripping `linalg` and `layout` and writing kernels against `std.gpu` only?*

Short version: **the libs the letter assumed were "MAX runtime, target-agnostic" are actually Mojo compiler runtime, pulled in unconditionally by every Mojo binary including hello-world.** Stripping `linalg`/`layout` from the source does *not* change the dynamic-library footprint on Darwin/Metal. It does, however, change which `.mojopkg`s are touched at compile time — and that may matter independently for licensing.

## Method

`spikes/mojo-runtime/src/` contains five Mojo source files, each compiled to `.dylib` via `mojo build --emit shared-lib`:

| Tier | Imports beyond prelude | Builds with `--target-accelerator` |
|---|---|---|
| 0 | `std.math` only | no |
| 1 | + `std.algorithm` (vectorize/parallelize), `std.memory` (alloc) | no |
| 2 | + `std.gpu`, `std.gpu.host` — hand-rolled kernel, raw `UnsafePointer` args | metal:4 |
| 3 | + `layout` (TileTensor / row_major) | metal:4 |
| 4 | + `linalg.matmul[target="gpu"]` — current `packages/rag` baseline | metal:4 |

Build + inspection driver: [`spikes/mojo-runtime/build_and_inspect.sh`](../spikes/mojo-runtime/build_and_inspect.sh). On Darwin it uses `otool -L` plus a transitive walk that follows `@rpath/` references through the pixi env. On Linux it uses `ldd`.

Per-tier transitive dep dumps land at `spikes/mojo-runtime/build/tier{0..4}.deps.txt`. Cross-tier summary at [`spikes/mojo-runtime/build/SUMMARY.md`](../spikes/mojo-runtime/build/SUMMARY.md).

## Findings — Darwin / Metal

### 1. Dynamic-library footprint is identical across all five tiers.

Every tier — including the hello-world `tier0` that does nothing but `sqrt(x) * 2.0` — links the same set of dylibs:

```
@rpath/libKGENCompilerRTShared.dylib       (direct)
@rpath/libAsyncRTMojoBindings.dylib        (direct)
/usr/lib/libSystem.B.dylib                 (direct, system)
@rpath/libMSupportGlobals.dylib            (transitive via libKGEN)
@rpath/libAsyncRTRuntimeGlobals.dylib      (transitive via libAsyncRTMojoBindings)
+ Apple system frameworks (Foundation, CoreFoundation, IOKit, Metal, libobjc, libc++)
```

Four of the five "MAX runtime, target-agnostic" libs the letter cited (`libKGEN*`, `libAsyncRT*` ×2, `libMSupportGlobals`) are present in tier 0. They are not MAX-platform-specific. They are emitted by the Mojo compiler for any `--emit shared-lib` build.

`pixi/envs/default/share/max/modular.cfg` confirms this categorically. Its `[mojo-max]` section explicitly enumerates these libs as part of the **Mojo compiler's** linker invocation:

```
compilerrt_path = .../libKGENCompilerRTShared.dylib
mgprt_path      = .../libMGPRT.dylib
shared_libs     = .../libAsyncRTMojoBindings.dylib
```

(`MGPRT` = Mojo GPU Runtime; appears in the linker config but is dlopened lazily on first GPU use rather than direct-linked, so tier 0 doesn't show it. Tiers 2–4 don't show it at link time either — the loader resolves it at runtime when DeviceContext is constructed.)

### 2. `linalg.matmul` is statically embedded, not dynamically loaded.

Tier 4's `.dylib` is 113K vs tier 3's 67K — the +46K is `linalg::gemv::gemv_split_k` and supporting symbols, verified via `nm -gU` on the resulting binary. The compiler picked `gemv_split_k` because the `Optional(ctx)` matmul call shape on M4 Metal selected the tall-skinny GEMM kernel.

This means **MAX kernel code from `linalg` ends up baked into our binary at compile time**, not loaded from a separate redistributable runtime library. Whether that compiled output is itself a "Redistributable Component" under the MCL is a separate question from the dynamic-library bundling question raised in the letter.

### 3. `layout` adds 4K to the binary; no new dynamic deps.

Tier 3 is 67K vs tier 2's 63K. The `layout` package contributes very little when used minimally (a single TileTensor wrap + indexing), and zero new dynamic deps beyond what tier 2 already had.

## Findings — Linux / sm_80 (H100)

Verified on NVIDIA H100 80GB HBM3 via RunPod on 2026-04-18. Capture: [`docs/spike-mojo-runtime-linux-20260418T173556Z.txt`](spike-mojo-runtime-linux-20260418T173556Z.txt).

### 1. All five libs from the letter are linked to tier 1 (CPU-only, no GPU code).

Tier 1 uses `std.algorithm` (vectorize/parallelize) and `std.memory` (alloc). It has no `std.gpu` import, no `layout`, no `linalg`, and no `--target-accelerator`. Yet `ldd` on `tier1.so` shows:

```
/workspace/mojo-addon-examples/.pixi/envs/default/lib/libKGENCompilerRTShared.so
/workspace/mojo-addon-examples/.pixi/envs/default/lib/libAsyncRTMojoBindings.so
/workspace/mojo-addon-examples/.pixi/envs/default/lib/libAsyncRTRuntimeGlobals.so
/workspace/mojo-addon-examples/.pixi/envs/default/lib/libMSupportGlobals.so
/workspace/mojo-addon-examples/.pixi/envs/default/lib/libNVPTX.so
/workspace/mojo-addon-examples/.pixi/envs/default/lib/libstdc++.so.6
/workspace/mojo-addon-examples/.pixi/envs/default/lib/libgcc_s.so.1
+ system: linux-vdso.so.1, libc.so.6, libm.so.6, libdl.so.2, ld-linux-x86-64.so.2
```

**The letter's framing of `libNVPTX.so` as "the NVIDIA PTX driver wrapper" pulled in only by GPU code is empirically wrong.** It's linked unconditionally. The Mojo async runtime (libAsyncRTMojoBindings) and/or the compiler driver pulls it in regardless of whether the source has any GPU code. Tiers 2, 3, 4 add no new dynamic libs on top of tier 1.

### 2. Tier 0 (hello world, `std.math` only) is fully statically linked.

`ldd tier0.so` → `statically linked`. A Mojo binary that only does `sqrt(x) * 2.0` has no dynamic library dependencies on Linux. The compiler elides everything. First real runtime hit is tier 1, where `alloc` and `parallelize` force the Mojo runtime in.

This means the five libs above aren't "injected by the linker into every Mojo binary" — they're pulled in by specific Mojo-stdlib features (async runtime, memory allocation). But they are **not** pulled in by `std.gpu`, `layout`, or `linalg` specifically; they predate any GPU import.

### 3. Tier-vs-tier dep sets are byte-identical (tiers 1–4) on Linux.

Same story as Darwin: `linalg` and `layout` add zero new dynamic dependencies. `linalg`'s kernel code is statically embedded (tier 4 is 148K vs tier 3's 50K — ~98K of compiled sm_80 PTX/SASS from `linalg::gemv::gemv_split_k` or equivalent).

### 4. Linux binary sizes (for reference)

| Tier | Linux size | Darwin size |
|---|---|---|
| 0 | 15K | 16K |
| 1 | 48K | 61K |
| 2 | 46K | 63K |
| 3 | 50K | 67K |
| 4 | 148K | 113K |

Tier 4 is bigger on Linux — embedded sm_80 PTX text is larger than Metal's AIR bytecode for the same kernel.

## Licensing implications

Three observations, in increasing strength:

### a. The libs the letter named "MAX runtime" are Mojo compiler runtime.

`libKGEN*`, `libAsyncRT*`, `libMSupport*` are emitted by the Mojo compiler driver for any binary it builds, including binaries that touch only `std.mojopkg` (which is open source under Apache 2.0 + LLVM exceptions per Modular's March-2024 announcement). If these libs were strictly off-limits to redistribute, the open-source Mojo standard library would be functionally unshippable in compiled form, which contradicts Modular's own positioning.

The MCL itself (https://www.modular.com/legal/community) does not enumerate Redistributable Components — it defers to "the accompanying SDK documentation or materials." Public-facing docs do not enumerate them either. The Modular community license forum thread #662 (Joe L., Modular) confirms `tensor`, `kv_cache`, `register`, `max` packages are explicitly "not stdlib" but does not name the dylibs.

The letter's framing ("MAX runtime, target-agnostic") was a guess; the empirical evidence is that they are Mojo-runtime, target-agnostic, and the letter to Modular should be re-framed accordingly.

### b. `linalg` and `layout` are separate `.mojopkg`s outside `std.mojopkg`.

The pixi env at `lib/mojo/` ships `std.mojopkg`, `linalg.mojopkg`, `layout.mojopkg`, `tensor.mojopkg`, `kv_cache.mojopkg`, `nn.mojopkg`, `_cublas.mojopkg`, etc. as distinct files. Per the forum thread, `tensor`/`kv_cache`/`register`/`max` are confirmed closed-source. `linalg` and `layout` are not in `std.mojopkg` and are not on the explicit "not stdlib" list — license status is publicly **undocumented**.

The conservative reading: treat `linalg.mojopkg` and `layout.mojopkg` as MCL-covered Redistributable Components. Compiled output of these packages, statically linked into our `.dylib`, may then itself fall under the standalone-redistribution restriction the letter flagged.

### c. Tier 2 (raw kernel via `std.gpu` only) avoids `linalg` and `layout` entirely.

`examples/image/addon.mojo` already proves the pattern: `UnsafePointer`-based kernels with `ctx.enqueue_function` work without importing `layout` or `linalg`. We can rewrite `packages/rag/src/kernels.mojo` to follow this pattern — see "Refactor sketch" below. The result would touch only `std.mojopkg`, which is unambiguously Apache-2.0 + LLVM-exceptions.

The dynamic-library footprint would not change (still pulls in the Mojo compiler runtime libs), but the compiled-in kernel code would all be derived from open-source `std.mojopkg`, not from `linalg`/`layout`. The Modular conversation then becomes:

> "We compile open-source Apache-2.0 Mojo source against the Mojo compiler. The resulting binary links against Mojo's compiler runtime libraries. Are those runtime libs (libKGEN, libAsyncRT, libMSupport, libNVPTX, libMGPRT) Redistributable Components subject to the standalone restriction, or are they implicitly redistributable in the way GCC's libgcc / libstdc++ are under the GCC runtime exception?"

That's a much cleaner question to ask Modular than the original letter's, and it has a much higher chance of "yes" because the alternative (no) implies Mojo binaries can't ship.

## Refactor sketch — `packages/rag` without `linalg`/`layout`

Replace `linalg.matmul[target="gpu"]` with a hand-rolled tile-based GEMM. Replace TileTensor wrapping with raw `UnsafePointer` (template: `examples/image/addon.mojo:_gpu_kernel_grayscale`). Concretely:

1. Move `_matmul_cached` and `_search_cached` in `packages/rag/src/kernels.mojo` to call a new `_gpu_kernel_matmul` written against `std.gpu` only.
2. The kernel: standard tiled FP32 GEMM with `stack_allocation[..., AddressSpace.SHARED]` from `std.gpu.memory` for the per-block A/B tiles, `std.gpu.barrier` for sync.
3. Performance impact unknown — `linalg` dispatches to tuned kernels (TF32 tensor cores on H100, scalar Metal kernels on M4). The notes in `kernels.mojo` (lines 51–60) record an earlier attempt at hand-rolling on M4 that came out 1.7× slower at batch=64. On H100 the `linalg` path uses TF32 tensor cores (4× theoretical speedup over scalar FP32); a hand-rolled FP32 kernel without tensor-core intrinsics would be slower. Hand-rolling with `wmma`-style intrinsics from `std.gpu` is possible but a much larger lift.
4. **Recommended bench gate before committing to refactor:** prototype the hand-rolled kernel on tier 2's pattern, bench at `[1, 384] × [384, 10k]` (the canonical `packages/rag` shape) on H100. If we land within 2× of `linalg`'s 0.06 ms, the licensing win pays for itself. If we're 5–10× slower, the package's value proposition collapses and we stay on `linalg`.

A separate option not explored here: keep `linalg` and ask Modular for explicit clearance on the compiled-in kernel code. Smaller scope than the original letter.

## Next steps

1. **Verify on Linux/H100.** Run the same five-tier inspector on RunPod:
   ```bash
   scripts/runpod-launch.sh \
     --capture-to docs/spike-mojo-runtime-h100-$(date -u +%Y%m%dT%H%M%SZ).txt -- \
     "bash spikes/mojo-runtime/build_and_inspect.sh && cat spikes/mojo-runtime/build/tier*.deps.txt"
   ```
   Expected: tiers 0–4 share an identical Linux dep set including `libNVPTX.so`. (If `libNVPTX.so` only appears in tiers 2–4, that would refine the licensing argument — it'd then be a GPU-codegen-driven dep, not unconditional.)
2. **Verify by symbol search** that `libKGEN*` and `libAsyncRT*` contain the symbols our binaries actually call (not just opaque hooks). `nm -D` on the Linux side; we already have evidence on Darwin.
3. **Re-draft the Modular letter.** Re-frame around "Mojo compiler runtime libs needed for any compiled Mojo binary" instead of "MAX runtime libs." Drop the bundling question for `linalg`-derived kernels (defer until we know if Modular wants them stripped); keep the question for `libKGEN*` etc. Counter-proposal: ask Modular to publish a GCC-runtime-exception-style clarification.
4. **Optional kernel rewrite.** Only worth doing if either (a) Modular says no on the runtime libs *and* (b) the hand-rolled kernel benches within an acceptable factor of `linalg`. Bench the prototype before committing.

## Files

- [`spikes/mojo-runtime/src/tier0_cpu_only.mojo`](../spikes/mojo-runtime/src/tier0_cpu_only.mojo)
- [`spikes/mojo-runtime/src/tier1_cpu_simd.mojo`](../spikes/mojo-runtime/src/tier1_cpu_simd.mojo)
- [`spikes/mojo-runtime/src/tier2_gpu_raw.mojo`](../spikes/mojo-runtime/src/tier2_gpu_raw.mojo)
- [`spikes/mojo-runtime/src/tier3_gpu_layout.mojo`](../spikes/mojo-runtime/src/tier3_gpu_layout.mojo)
- [`spikes/mojo-runtime/src/tier4_gpu_linalg.mojo`](../spikes/mojo-runtime/src/tier4_gpu_linalg.mojo)
- [`spikes/mojo-runtime/build_and_inspect.sh`](../spikes/mojo-runtime/build_and_inspect.sh)
- [`spikes/mojo-runtime/build/SUMMARY.md`](../spikes/mojo-runtime/build/SUMMARY.md) — Darwin tier-vs-tier deps table
