# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

High-performance Node.js addon examples written in Mojo and bridged to JS through [napi-mojo](https://github.com/codetalcott/napi-mojo). Each addon showcases SIMD `vectorize()` + `parallelize()` on CPU, and (for most) a GPU kernel via MAX's `DeviceContext`. Benchmarks target both M4 Metal and NVIDIA H100 (via RunPod/Lambda).

The repo is structured as three cohabiting things:

- **`examples/`** — one directory per kernel (matmul, simd-search, stats, image, wyhash, rag-demo). Each is a self-contained Mojo addon + JS demo/bench/test.
- **`packages/rag/`** — `@qkstat/rag`, a distributed Node package (GPU exact-retrieval primitives: `loadMatrixGpu` / `matmulHandle` / `searchHandle` / `releaseMatrixGpu` + `GpuIndex`). Has its own build, platform sub-packages under `npm/`, and Jest tests.
- **`packages/embed/`** — `@qkstat/embed`, MiniLM-L6-v2 embeddings on H100 via MAX + Python interop. Productized from the embedding-kernel spike (2026-04-17 GO verdict); composes with `packages/rag` at runtime in one Node process. Historical spike log at [`docs/embedding-kernel-spike-findings.md`](docs/embedding-kernel-spike-findings.md).

The `@qkstat` landing page at [qkstat.dev](https://qkstat.dev) is built from a separate repo (`qkstat-site`, Eleventy + Fly.io + Cloudflare). When a new package publishes or benchmark numbers change, `src/_data/packages.yaml` in that repo is the single maintenance surface — add an entry or flip `status: pre-release` → `published`.

## Common commands

The Mojo toolchain comes from pixi; Node scripts run under the pixi environment too.

```bash
npm install                      # install napi-mojo + JS deps
pixi install                     # install Mojo/MAX

# Build everything under examples/ (one .node per example)
npm run build:all

# Single example (pattern: build:<name>, demo:<name>)
npm run build:matmul
npm run demo:matmul
pixi run bash examples/matmul/build.sh        # equivalent to build:matmul

# Cached (persistent-buffer) variants live beside the one-shot builds
pixi run bash examples/matmul/build_cached.sh
pixi run bash examples/simd-search/build_cached.sh
pixi run bash examples/stats/build_cached.sh
pixi run bash examples/image/build_cached.sh

# packages/rag — built and tested separately
pixi run bash packages/rag/build.sh
(cd packages/rag && npm test)                 # Jest

# packages/embed — built separately; depends on packages/rag being built first
pixi run bash packages/rag/build.sh && pixi run bash packages/embed/build.sh
pixi run node packages/embed/test-roundtrip.js   # Gate F4 correctness
pixi run node packages/embed/demo.js             # GPU end-to-end with packages/rag
pixi run node packages/embed/bench.js            # MS-MARCO warm-path benchmarks

# Correctness tests (root): runs all per-example test.js files
npm test

# Run a single example test directly
node examples/matmul/test.js
node examples/simd-search/test_cached.js      # cached-variant tests
```

## GPU target flags

Each `build.sh` auto-selects a `--target-accelerator` and exposes an env var to override. Defaults:

- Darwin arm64 → `metal:4`
- Linux x86_64 → `sm_90` for most (H100/H200); `sm_80` for `packages/rag` (NVIDIA baseline, PTX forward-compat covers 80/86/89/90/100+ via driver JIT so one binary ships everywhere)

Override per addon:

- `STATS_ACCEL`, `SEARCH_ACCEL`, `IMAGE_ACCEL`, `MATMUL_ACCEL` — per-example (empty string = CPU-only)
- `QKSTAT_RAG_ACCEL` — for `packages/rag`
- `EMBED_ACCEL` — for `packages/embed`

On Linux x86_64, builds add `--mcpu haswell` to avoid AVX-512 instructions that break on older runners (e.g. GitHub Actions). AVX-512 is still used at runtime on hosts that support it for Mojo's SIMD width selection — the `--mcpu` flag only constrains the baseline.

## Cloud benchmark infrastructure

`scripts/runpod-launch.sh` + `scripts/bootstrap.sh` are the canonical path for H100 runs. Pods use a persistent Network Volume that caches pixi env + model weights so each session starts in ~30 s. `trap EXIT` in the launcher guarantees pod termination. See `docs/cloud-benchmark-runbook.md` for the ~30-minute reproduction flow (~$1/run). Phase-specific runners: `scripts/runpod-bench-3{b,c,d}.sh`.

## Architecture

### The napi-mojo bridge pattern

Every addon follows the same shape:

```
example-name/
  addon.mojo           # SIMD/parallel kernels + N-API callbacks
  addon_cached.mojo    # (sometimes) persistent-buffer variant — handle-based API
  build.sh             # Mojo → .node shared library (platform-aware)
  example.js           # benchmark / demo
  test.js              # correctness tests (vs JS reference)
  README.md            # per-example benchmark tables + caveats
```

Kernels use napi-mojo's framework: `napi.types`, `napi.framework.js_typedarray`, `napi.framework.args`, `napi.framework.register.ModuleBuilder`. JS passes pre-allocated output buffers to avoid allocation in the hot path. `JsTypedArray.data_ptr(env).unsafe_bitcast[Float64]()` gives zero-copy access to V8 memory. `parallelize()` requires `init_async_runtime()` to be called once at module load.

### One-shot vs cached (persistent-buffer) API

The benchmarking story the repo is *telling* is that single-call GPU APIs are PCIe-bound for low-arithmetic-intensity kernels. The fix is the **cached/handle API**: upload once (`loadGpu`), query many times (`countByteHandle`, `matmulHandle`, etc.), release (`releaseGpu`). This is where the 1000×+ speedups live (Phase 3a countByte, Phase 3c matmul). Details per-addon are in `README.md`; the root `README.md` has the cross-addon summary tables that are kept current with each phase.

When adding a new GPU kernel, prefer the cached pattern if the input is large and queried repeatedly. The `packages/rag` module is the cleanest reference for the pattern.

### `packages/rag` specifics

- Source: [`packages/rag/src/lib.mojo`](packages/rag/src/lib.mojo) (N-API surface) + `packages/rag/src/kernels.mojo`. Built with `-I src -I "$NAPI_SRC"` (see [`scripts/napi-include.sh`](scripts/napi-include.sh)) so `lib.mojo` can `from linalg import ...` from MAX.
- Loader: [`packages/rag/index.js`](packages/rag/index.js) tries the platform sub-package (`@qkstat/rag-darwin-arm64` / `@qkstat/rag-linux-x64`), then falls back to the local `build/rag.node`. The sub-packages live under `packages/rag/npm/`; bundling is `bash scripts/bundle-libs.sh`.
- Tests: Jest under `packages/rag/tests/`. They require the addon to be built first.

### `packages/embed` specifics

Composition model: `packages/embed/build/embed.node` (MAX Python interop for MiniLM) and `packages/rag/build/rag.node` (Mojo N-API matmul) load into the **same Node process** with **separate CUDA contexts**. This is the "kernel-factory" thesis the spike validated. The `embed_batch_from_addrs` path in `packages/embed/embed.py` is a raw-address memmove; DLPack zero-copy is a deferred optimization. [`docs/embedding-kernel-spike-findings.md`](docs/embedding-kernel-spike-findings.md) is the day-by-day execution log from the original spike (archived after productization).

The Mojo addon imports its Python module by inserting `packages/embed` into `sys.path` at runtime — handles both pod (`/workspace/mojo-addon-examples/packages/embed`) and laptop (relative `packages/embed`) paths. See [`packages/embed/src/embed.mojo`](packages/embed/src/embed.mojo) `_import_embed_module`.

The repo root's `pixi.toml` includes `transformers`, `safetensors`, etc. specifically for `packages/embed`'s `BertPipelineModel` loading. Don't strip those.

## Notable constraints

- **Mojo version pin** lives in `pixi.toml`. As of 2026-08-12 this is **stable Mojo 1.0.0** (`max = "==26.5.0"`, from the stable `https://conda.modular.com/max/` channel) rather than a nightly, matching napi-mojo 0.7.0, which pins stable releases for exactly the reason below. `scripts/update-mojo-version.sh` rewrites only the version line, so repoint `channels` too if you ever move back to nightlies.
  The pin must track whatever Mojo version the installed napi-mojo was migrated to — the framework ships Mojo *source*, so a mismatch surfaces as compile errors inside `node_modules/napi-mojo/src`, not in this repo's code. Tracking a stable release on both sides is what makes that coupling manageable.
- **`pixi.lock` is git-ignored here** (`.gitignore`), so builds are not reproducible across machines from the repo alone — the pin in `pixi.toml` is the only record. napi-mojo tracks its lock deliberately: a lock-less bump there took its Nightly Canary down for two weeks. Worth reconsidering.
- **`.last-good-nightly` is a hand-maintained note, not machinery.** Nothing in the repo reads or writes it — no script, workflow, or doc — so it drifts silently and has done so more than once. Despite the name it now records the last *verified toolchain*, nightly or stable (currently `26.5.0`, i.e. Mojo 1.0.0). **`pixi.toml` is the authoritative pin**; if the two ever disagree, `pixi.toml` is right.
- **GPU kernel parameters must be fixed-width** (Mojo 26.6+). `Int`/`UInt` no longer conform to `DevicePassable`, so a kernel launched via `ctx.enqueue_function[...]` cannot take an `Int` size argument — it fails with `constraint failed: Int and UInt do not conform to DevicePassable`. Every kernel here takes its length as `Int64` and converts back with `var n = Int(n_i64)` on the first line of the body, which keeps the indexing logic unchanged. The host side (`enqueue_create_buffer`, `ceildiv`, grid math) is unaffected and still uses `Int`.
- **Apple Silicon GPU benchmarks** require Xcode's Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`). Without it, GPU builds on Darwin fail at link time.
- **MAX on M4 is currently CPU-only for `packages/embed`** — `Accelerator()` init returns "Not implemented for device: Apple M4". All embed GPU iteration happens on RunPod.
- **Node version**: `engines.node >=22.12` in `packages/rag`. Root examples also assume that.
- **Build outputs are git-ignored** (`build/*.node`, fixtures). Don't commit them.

## napi-mojo linkage

All examples depend on napi-mojo installed via npm, which provides the framework sources that `-I` pulls into Mojo builds. napi-mojo is a *source* framework (the node-addon-api model): `require('napi-mojo')` returns paths, not a compiled addon, and `.include` is the directory to pass to `mojo build -I`. Every `build.sh` resolves it through [`scripts/napi-include.sh`](scripts/napi-include.sh) rather than hardcoding `node_modules/napi-mojo/src`, so hoisting and `npm link` both work. The compiled demo addon, if you want it, is `require('napi-mojo/demo')`.

For local iteration against an unreleased napi-mojo:

```bash
cd /path/to/napi-mojo && npm link
cd /path/to/mojo-addon-examples && npm link napi-mojo
```

`npm install` reverts to the published package.
