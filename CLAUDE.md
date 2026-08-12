# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

High-performance Node.js addon examples written in Mojo and bridged to JS through [napi-mojo](https://github.com/codetalcott/napi-mojo). Each addon showcases SIMD `vectorize()` + `parallelize()` on CPU, and (for most) a GPU kernel via MAX's `DeviceContext`. Benchmarks target both M4 Metal and NVIDIA H100 (via RunPod/Lambda).

The repo is structured as three cohabiting things:

- **`examples/`** — one directory per kernel (matmul, simd-search, stats, image, wyhash, rag-demo). Each is a self-contained Mojo addon + JS demo/bench/test.
- **`packages/retrieve/`** — `@qkstat/retrieve`, a distributed Node package (GPU exact-retrieval primitives: `loadMatrixGpu` / `matmulHandle` / `searchHandle` / `releaseMatrixGpu` + `GpuIndex`). Has its own build, platform sub-packages under `npm/`, and Jest tests.
- **`packages/embed/`** — `@qkstat/embed`, MiniLM-L6-v2 embeddings on H100 via MAX + Python interop. Productized from the embedding-kernel spike (2026-04-17 GO verdict); composes with `packages/retrieve` at runtime in one Node process. Historical spike log at [`docs/embedding-kernel-spike-findings.md`](docs/embedding-kernel-spike-findings.md).

The `@qkstat` landing page at [qkstat.dev](https://qkstat.dev) is built from a separate repo (`qkstat-site`, Eleventy + Fly.io + Cloudflare). When a new package publishes or benchmark numbers change, `src/_data/packages.yaml` in that repo is the single maintenance surface — add an entry or flip `status: pre-release` → `published`.

## Common commands

The Mojo toolchain comes from pixi; Node scripts run under the pixi environment too.

```bash
npm install                      # install napi-mojo + JS deps
pixi install                     # install Mojo/MAX

# Verify EVERYTHING locally — the comprehensive gate, run this before pushing.
# Builds all 11 targets (5 one-shot + 4 cached + retrieve + embed) and runs all 21
# suites, including the GPU-execution tests that CI structurally cannot run.
bash scripts/verify-all.sh
bash scripts/verify-all.sh --skip-embed        # faster; skips the slowest build
bash scripts/verify-all.sh --with-embed-test   # + embed roundtrip (needs weights)

# Build the five ONE-SHOT example addons. Despite the name this is not
# everything — it excludes the four cached variants and both packages, which
# have their own scripts below.
npm run build:all

# Single example (pattern: build:<name>, demo:<name>)
npm run build:matmul
npm run demo:matmul
pixi run bash examples/matmul/build.sh        # equivalent to build:matmul

# Cached (persistent-buffer) variants live beside the one-shot builds
npm run build:cached                          # all four
npm run build:matmul-cached                   # or one at a time
pixi run bash examples/matmul/build_cached.sh # equivalent

# packages/retrieve — built and tested separately
npm run build:retrieve                        # = bash packages/retrieve/build.sh
(cd packages/retrieve && npm test)                 # Jest

# packages/embed — built separately; depends on packages/retrieve being built first
npm run build:retrieve && npm run build:embed
pixi run node packages/embed/test-roundtrip.js   # Gate F4 correctness
pixi run node packages/embed/demo.js             # GPU end-to-end with packages/retrieve
pixi run node packages/embed/bench.js            # MS-MARCO warm-path benchmarks

# Correctness tests (root): the five per-example test.js files, CPU paths only.
# NOT a full check — it skips every cached variant, packages/retrieve, and
# packages/embed, and touches no GPU code at all. It is also &&-chained, so it
# stops at the first failing addon and hides the rest. Use verify-all.sh above
# to actually verify the repo; `npm test` is the CI smoke subset.
#
# CI compiles the cached variants and both packages, and additionally runs the
# simd-search/stats/image cached suites on the macOS job. matmul and
# packages/retrieve are NOT run in CI — the runner's Metal device cannot
# execute their linalg-backed kernels. See "CI coverage" below.
npm test

# Run a single example test directly
node examples/matmul/test.js
node examples/simd-search/test_cached.js      # cached-variant tests
```

## GPU target flags

Each `build.sh` auto-selects a `--target-accelerator` and exposes an env var to override. Defaults:

- Darwin arm64 → `metal:4`
- Linux x86_64 → `sm_90` for most (H100/H200); `sm_80` for `packages/retrieve` (NVIDIA baseline, PTX forward-compat covers 80/86/89/90/100+ via driver JIT so one binary ships everywhere)

Override per addon:

- `STATS_ACCEL`, `SEARCH_ACCEL`, `IMAGE_ACCEL`, `MATMUL_ACCEL` — per-example
- `QKSTAT_RETRIEVE_ACCEL` — for `packages/retrieve`
- `EMBED_ACCEL` — for `packages/embed`

**Setting these to the empty string does not give you a CPU-only build.** It only suppresses the `--target-accelerator` flag, and with that flag absent Mojo falls back to the *host* accelerator — on an M-series Mac, `mojo build --print-effective-target` reports `--target-accelerator metal:4-metal4`. There is no "no accelerator" flag. Any addon whose source contains GPU kernels therefore always invokes a GPU compiler, so on Darwin it needs the Metal Toolchain (see Notable constraints) no matter how these vars are set. `matmul/addon.mojo` and `wyhash/addon.mojo` are the only GPU-free sources in the repo; every other addon — including all four `addon_cached.mojo` variants — has GPU code. Use these vars to *retarget* a build (e.g. `sm_80` instead of `sm_90`), not to opt out of one.

On Linux x86_64, builds add `--mcpu haswell` to avoid AVX-512 instructions that break on older runners (e.g. GitHub Actions). AVX-512 is still used at runtime on hosts that support it for Mojo's SIMD width selection — the `--mcpu` flag only constrains the baseline.

## Cloud benchmark infrastructure

`scripts/runpod-launch.sh` + `scripts/bootstrap.sh` are the canonical path for H100 runs. `trap EXIT` in the launcher guarantees pod termination, backed by a RunPod-side `terminateAfter` (default 2 h). Phase-specific runners: `scripts/runpod-bench-3{b,c,d}.sh`.

**There is currently no Network Volume.** The account had none as of 2026-08-12 — a launch against the configured `RUNPOD_VOLUME_ID` failed with `Network volume "vdmny3q3os" not found`, and the API reports zero volumes. So the "~30 s warm start" this tooling was built around does not apply, and `docs/cloud-benchmark-runbook.md`'s ~$1/30-minute figures assume a warm volume that no longer exists.

Until one is recreated and reseeded, use `--no-volume`, which does a self-contained cold bootstrap (installs pixi + node, clones the public repo, no volume credentials needed):

```bash
bash scripts/runpod-launch.sh --no-volume --capture-to docs/<name>.txt -- \
  "npm install --no-audit --no-fund && bash scripts/verify-all.sh --with-embed-test"
```

`npm install` is required in the command: the cold bootstrap installs the *toolchains* but never runs it, and `verify-all.sh` will otherwise stop at its napi-mojo preflight. Budget ~25–35 min and ~$2 — nothing caches between runs, so every session re-downloads the MAX toolchain and model weights.

## CI coverage

`.github/workflows/test.yml` runs on `ubuntu-latest` always, plus `macos-latest` on PRs to `main`. The repo is **public**, so standard-runner minutes are free and unmetered — the budget to protect is PR feedback latency, not dollars.

CI **compiles everything** (`build:all`, `build:cached`, `build:retrieve`, `build:embed`) on both runners, and additionally **executes three GPU suites on macOS only**, guarded by `if: runner.os == 'macOS'`. What runs where is a capability limit, not a cost decision.

**`macos-latest` executes Metal, but only partly.** Measured by `.github/workflows/metal-probe.yml`, which runs every suite independently (run 31630265503, 2026-08-12):

| Suite | Kernel style | Runner |
| --- | --- | --- |
| `simd-search` | hand-rolled `std.gpu` | pass — 220 cases |
| `stats` | hand-rolled `std.gpu` | pass — 208 cases |
| `image` | hand-rolled `std.gpu` | pass — 210 cases |
| `matmul` | `linalg`/`layout` | **fail** |
| `packages/retrieve` | `linalg`/`layout` | **fail** |

The runner's Metal device runs hand-rolled `std.gpu` kernels but not `linalg`-backed ones — tiers 2 vs 3/4 in [`docs/mojo-runtime-isolation-spike-findings.md`](docs/mojo-runtime-isolation-spike-findings.md). `packages/retrieve` shows the split cleanly: `loadMatrixGpu` and every lifecycle/error-path test pass, while all four tests invoking `matmulHandle`/`searchHandle` fail. **Allocation works; computation does not.** Both `matmul` and `retrieve` pass on real NVIDIA, so this is a runner limitation, not a kernel bug.

Re-run the probe (Actions → "Metal GPU probe (manual)") if runner images change. It builds `build:all` **and** `build:cached` on purpose: `stats/test_cached.js` and `image/test_cached.js` load both the one-shot and cached addons, and omitting the former makes them die on `stats.node not found` — which reads as a GPU failure and is not one.

**Two things to keep in mind.** `ubuntu-latest` has no NVIDIA GPU, so it can never run these — permanent. And `macos-latest` runs only on PRs targeting `main`, so the GPU suites gate PRs but **not** pushes to `main` and **not** the weekly cron. A regression arriving by a route that skips the macOS job will not be caught in CI. Treat CI's GPU coverage as a free subset, never as the gate.

Each suite is its own workflow step, never an `&&` chain — chaining lets the first failure hide the rest, which is how an earlier version of this hid three suites behind `matmul`.

The complete gate stays `scripts/verify-all.sh` locally and `scripts/verify-gpu-h100.sh` on a pod. Those are also the only authority for the NVIDIA paths, since no hosted runner has an NVIDIA GPU — CI verifies Metal, production ships CUDA.

Compile-only coverage still earns its place: `packages/retrieve` sat uncompiled from 2026-04-17 to 2026-08-12 and silently missed an entire napi-mojo + Mojo 1.0.0 migration, because nothing in CI ever built it. CI is also the only place the Linux/`sm_90` cross-compile that ships to RunPod is exercised — a Darwin laptop cannot produce that artifact.

Docs-only changes skip CI entirely (`paths-ignore` covers `**.md` and `docs/**`).

## Architecture

### The napi-mojo bridge pattern

Every addon follows the same shape:

```text
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

When adding a new GPU kernel, prefer the cached pattern if the input is large and queried repeatedly. The `packages/retrieve` module is the cleanest reference for the pattern.

### `packages/retrieve` specifics

- Source: [`packages/retrieve/src/lib.mojo`](packages/retrieve/src/lib.mojo) (N-API surface) + `packages/retrieve/src/kernels.mojo`. Built with `-I src -I "$NAPI_SRC"` (see [`scripts/napi-include.sh`](scripts/napi-include.sh)) so `lib.mojo` can `from linalg import ...` from MAX.
- Loader: [`packages/retrieve/index.js`](packages/retrieve/index.js) tries the platform sub-package (`@qkstat/retrieve-darwin-arm64` / `@qkstat/retrieve-linux-x64`), then falls back to the local `build/retrieve.node`. The sub-packages live under `packages/retrieve/npm/`; bundling is `bash scripts/bundle-libs.sh`.
- Tests: Jest under `packages/retrieve/tests/`. They require the addon to be built first.

### `packages/embed` specifics

Composition model: `packages/embed/build/embed.node` (MAX Python interop for MiniLM) and `packages/retrieve/build/retrieve.node` (Mojo N-API matmul) load into the **same Node process** with **separate CUDA contexts**. This is the "kernel-factory" thesis the spike validated. The `embed_batch_from_addrs` path in `packages/embed/embed.py` is a raw-address memmove; DLPack zero-copy is a deferred optimization. [`docs/embedding-kernel-spike-findings.md`](docs/embedding-kernel-spike-findings.md) is the day-by-day execution log from the original spike (archived after productization).

The Mojo addon imports its Python module by inserting `packages/embed` into `sys.path` at runtime — handles both pod (`/workspace/mojo-addon-examples/packages/embed`) and laptop (relative `packages/embed`) paths. See [`packages/embed/src/embed.mojo`](packages/embed/src/embed.mojo) `_import_embed_module`.

The repo root's `pixi.toml` includes `transformers`, `safetensors`, etc. specifically for `packages/embed`'s `BertPipelineModel` loading. Don't strip those.

## Notable constraints

- **Mojo version pin** lives in `pixi.toml`. As of 2026-08-12 this is **stable Mojo 1.0.0** (`max = "==26.5.0"`, from the stable `https://conda.modular.com/max/` channel) rather than a nightly, matching napi-mojo 0.7.0, which pins stable releases for exactly the reason below. `scripts/update-mojo-version.sh` rewrites only the version line, so repoint `channels` too if you ever move back to nightlies.
  The pin must track whatever Mojo version the installed napi-mojo was migrated to — the framework ships Mojo *source*, so a mismatch surfaces as compile errors inside `node_modules/napi-mojo/src`, not in this repo's code. Tracking a stable release on both sides is what makes that coupling manageable.
- **`pixi.lock` is git-ignored here** (`.gitignore`), so builds are not reproducible across machines from the repo alone — the pin in `pixi.toml` is the only record. napi-mojo tracks its lock deliberately: a lock-less bump there took its Nightly Canary down for two weeks. Worth reconsidering.
- **`.last-good-nightly` is a hand-maintained note, not machinery.** Nothing in the repo reads or writes it — no script, workflow, or doc — so it drifts silently and has done so more than once. Despite the name it now records the last *verified toolchain*, nightly or stable (currently `26.5.0`, i.e. Mojo 1.0.0). **`pixi.toml` is the authoritative pin**; if the two ever disagree, `pixi.toml` is right.
- **GPU kernel parameters must be fixed-width** (Mojo 26.6+). `Int`/`UInt` no longer conform to `DevicePassable`, so a kernel launched via `ctx.enqueue_function[...]` cannot take an `Int` size argument — it fails with `constraint failed: Int and UInt do not conform to DevicePassable`. Every kernel here takes its length as `Int64` and converts back with `var n = Int(n_i64)` on the first line of the body, which keeps the indexing logic unchanged. The host side (`enqueue_create_buffer`, `ceildiv`, grid math) is unaffected and still uses `Int`.
- **Xcode's Metal Toolchain is required to build on Darwin at all** (`xcodebuild -downloadComponent MetalToolchain`, ~688 MB; check with `xcodebuild -showComponent MetalToolchain`). This is not benchmark-only: without it, every addon containing GPU kernels fails to *compile* (`error: Metal Compiler failed to compile metallib` / `failed to run the pass manager`) — not at link time, and with no way to opt out via the `*_ACCEL` vars (see GPU target flags). Only `matmul` and `wyhash` build without it, so a Mac missing this component silently ends up with stale `.node` files for everything else while `npm run build:all` still appears to make progress.
- **MAX on M4 is currently CPU-only for `packages/embed`** — `Accelerator()` init returns "Not implemented for device: Apple M4". All embed GPU iteration happens on RunPod.
- **Node version**: `engines.node >=22.12` in `packages/retrieve`. Root examples also assume that.
- **Build outputs are git-ignored** (`build/*.node`). Don't commit them. "Fixtures" is *not* a blanket rule despite what this line used to say: `.gitignore` covers only `examples/rag-demo/fixtures/*.bin`, while `examples/matmul/fixtures/matmul_dyn.onnx` is tracked. `packages/embed/fixtures/` is neither ignored nor committed — it is generated by `packages/embed/reference.js`, so a fresh clone has none and `test-roundtrip.js` fails until it runs. `scripts/verify-all.sh --with-embed-test` now generates them when absent.

## napi-mojo linkage

All examples depend on napi-mojo installed via npm, which provides the framework sources that `-I` pulls into Mojo builds. napi-mojo is a *source* framework (the node-addon-api model): `require('napi-mojo')` returns paths, not a compiled addon, and `.include` is the directory to pass to `mojo build -I`. Every `build.sh` resolves it through [`scripts/napi-include.sh`](scripts/napi-include.sh) rather than hardcoding `node_modules/napi-mojo/src`, so hoisting and `npm link` both work. The compiled demo addon, if you want it, is `require('napi-mojo/demo')`.

For local iteration against an unreleased napi-mojo:

```bash
cd /path/to/napi-mojo && npm link
cd /path/to/mojo-addon-examples && npm link napi-mojo
```

`npm install` reverts to the published package.
