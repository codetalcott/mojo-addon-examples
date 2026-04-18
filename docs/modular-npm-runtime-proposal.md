# Proposal — publish `@modular/mojo-runtime-*` npm packages

## Context

The Mojo compiler emits `.node` / `.dylib` / `.so` binaries that depend on a small set of shared libraries:

- `libKGENCompilerRTShared` — compiler runtime (MLIR IR codegen)
- `libAsyncRTMojoBindings`, `libAsyncRTRuntimeGlobals` — async runtime
- `libMSupportGlobals` — support runtime
- `libMGPRT` — Mojo GPU runtime (lazy-loaded)
- `libNVPTX` — CUDA/PTX driver glue
- `libstdc++`, `libgcc_s` — C++ stdlib (Linux-bundled)

These are Mojo compiler runtime, not MAX-platform libs — a hello-world Mojo binary that uses `std.algorithm.parallelize` links the full set even with no GPU code. Any developer shipping a Mojo-compiled Node addon needs these at runtime.

Today, the only way to get them is `pixi install max` (or `pip install modular`), which installs the whole MAX SDK (GBs). For Node addon authors, that's unacceptable install UX. The alternative — bundling the libs inside the addon package — has unresolved licensing status under the MCL's standalone-redistribution clause.

**This proposal: Modular publishes the Mojo compiler runtime as a small set of npm packages, analogous to `@esbuild/linux-x64`. Wm writes the packaging and CI; Modular reviews and holds publish credentials. Estimated effort: ~1 engineering week on Wm's side, ~2 hours of Modular review + npm token provisioning.**

## Why this matters for Modular

This is the Node-ecosystem analog of Modular's existing `pip install modular` distribution — closing a structural gap that today blocks Mojo-compiled addons from reaching Node developers.

- **Install UX gates everything.** A Mojo-compiled Node addon today requires users to first `pixi install max` (multi-GB SDK). There is no "just `npm install`" path. Every would-be Mojo-Node author hits this wall. A runtime package is the one missing piece between Mojo and the Node ecosystem.
- **The bindings layer already exists.** [napi-mojo](https://github.com/codetalcott/napi-mojo) — ~140 exported functions, ~620 Jest tests, published to npm since early 2026 — is the framework for compiling Mojo into Node `.node` addons. The runtime-distribution story this proposal fills is what unblocks it at scale.
- **Adoption compounds on existing pilots.** [`@qkstat/rag`](../packages/rag/) (sub-100 µs exact RAG at recall=1.0 on H100) and [`@qkstat/embed`](../packages/embed/) (MiniLM inference composed with rag in one Node process) are working proof points. A runtime package turns each into a live "Mojo made this faster in Node" demo shipping with the kind of perf numbers that are useful on Modular's marketing surface — and makes it easy for the next author to follow the pattern.
- **Parity with the Python story.** `pip install modular` is already the canonical Python-side onboarding. `npm install @modular/mojo-runtime` gives Node the same shape. Symmetric across the two biggest AI/ML scripting ecosystems.
- **Low downside.** ~1 week of external engineering, zero Mojo/MAX/compiler changes, reversible deprecation path. If adoption stalls at v0.1, the subpackages can be deprecated without Modular having committed to long-tail platforms.

## The proposal at a glance

- **Parent:** `@modular/mojo-runtime` — ~2 KB loader with no code of its own.
- **Platform subpackages:** `@modular/mojo-runtime-linux-x64`, `@modular/mojo-runtime-darwin-arm64` (initial; more to follow).
- **Contents per subpackage:** the 6–8 runtime `.so` / `.dylib` files from the pixi env. Total ~8 MB on Darwin, ~15 MB on Linux (libNVPTX + glibc compat shims).
- **Versioning:** 1:1 with Mojo toolchain versions. `@modular/mojo-runtime@0.26.3` ships the runtime from Mojo 0.26.3.
- **Consumer pattern:** `require('@modular/mojo-runtime')` returns `{ libDir, libNames, version }`. Node addons either rpath to `libDir` at build time or `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` at load time.
- **Not in scope:** the Mojo compiler driver, `stdlib.mojopkg`, MAX (`linalg`, `layout`, `tensor`, `kv_cache`, `nn`, etc.), Python interop, CLI tools. **Runtime libraries only.**

## What we ship — precise contents

### Package tree

```
@modular/mojo-runtime/                   # parent loader, pure JS
├── package.json                         # optionalDependencies on all platform pkgs
├── index.js                             # platform resolver → subpackage
├── README.md
└── LICENSE                              # MCL or whatever Modular chooses

@modular/mojo-runtime-linux-x64/         # platform subpackage, binaries
├── package.json                         # os:["linux"], cpu:["x64"]
├── lib/
│   ├── libKGENCompilerRTShared.so
│   ├── libAsyncRTMojoBindings.so
│   ├── libAsyncRTRuntimeGlobals.so
│   ├── libMSupportGlobals.so
│   ├── libMGPRT.so
│   ├── libNVPTX.so
│   ├── libstdc++.so.6                   # bundled for Ubuntu 20.04 compat
│   └── libgcc_s.so.1
└── README.md

@modular/mojo-runtime-darwin-arm64/      # Apple Silicon variant
├── package.json                         # os:["darwin"], cpu:["arm64"]
├── lib/
│   ├── libKGENCompilerRTShared.dylib
│   ├── libAsyncRTMojoBindings.dylib
│   ├── libAsyncRTRuntimeGlobals.dylib
│   ├── libMSupportGlobals.dylib
│   ├── libMGPRT.dylib
│   └── libNVPTX.dylib                   # Metal equivalent; name TBD
└── README.md
```

### Parent `package.json` shape

```json
{
  "name": "@modular/mojo-runtime",
  "version": "0.26.3",
  "description": "Mojo compiler runtime libraries for Node.js addons built with Mojo",
  "main": "index.js",
  "engines": { "node": ">=18" },
  "license": "…",
  "optionalDependencies": {
    "@modular/mojo-runtime-linux-x64": "0.26.3",
    "@modular/mojo-runtime-darwin-arm64": "0.26.3"
  },
  "repository": "github:modular/modular"
}
```

### Parent `index.js` shape (~30 lines)

```js
const path = require('path');
const PLATFORMS = {
  'linux-x64': '@modular/mojo-runtime-linux-x64',
  'darwin-arm64': '@modular/mojo-runtime-darwin-arm64',
};
const key = `${process.platform}-${process.arch}`;
const pkg = PLATFORMS[key];
if (!pkg) throw new Error(`@modular/mojo-runtime: no prebuilt for ${key}`);
const pkgPath = path.dirname(require.resolve(`${pkg}/package.json`));
module.exports = {
  libDir: path.join(pkgPath, 'lib'),
  libNames: require(`${pkg}/package.json`).libNames,
  version: require(`${pkg}/package.json`).version,
};
```

### Platform subpackage `package.json` shape

```json
{
  "name": "@modular/mojo-runtime-linux-x64",
  "version": "0.26.3",
  "os": ["linux"],
  "cpu": ["x64"],
  "files": ["lib/", "README.md"],
  "libNames": [
    "libKGENCompilerRTShared.so", "libAsyncRTMojoBindings.so", "…"
  ]
}
```

## How we build — Wm writes this

**Source-of-truth.** The existing pipeline is unchanged: `pixi install` materializes the runtime libs in `.pixi/envs/default/lib/`. Our packaging reads from there. No changes to Mojo compiler, Mojo stdlib, or MAX.

**Packaging script** (new, at `tools/pack-mojo-runtime-npm/`, to live in either `modular/modular` or a separate repo of Modular's choosing). Wm writes. Outline:

1. Take a pixi-installed MAX env directory as input.
2. For each target (linux-x64, darwin-arm64, …):
   - Copy the listed libs into `lib/`.
   - Linux: run `patchelf --set-rpath '$ORIGIN'` on each bundled lib so inter-lib resolution stays inside the package.
   - Darwin: run `install_name_tool -id @rpath/<name> <file>` and `install_name_tool -change` to rewrite inter-lib paths to `@rpath/`-relative.
   - Emit `package.json` with pinned version (from `pixi list max`).
3. Emit the parent package with matching version + `optionalDependencies` pinned to same.
4. Emit `pnpm-workspace.yaml` or similar for atomic publish.

This is ~150 lines of shell+Node. Existing template: [`../packages/rag/scripts/bundle-libs.sh`](../packages/rag/scripts/bundle-libs.sh) already implements the Linux rpath step for our own addon — Wm adapts it.

**CI pipeline** (new workflow file). Wm writes. Runs on tag push. Matrix over `{linux-x64, darwin-arm64}`:

1. `pixi install` at the tagged MAX version.
2. Run the packaging script.
3. `npm publish` each subpackage, then the parent.

The CI lives wherever Modular wants — either `modular/modular` as a new workflow, or a new `modular/mojo-runtime-npm` repo (Wm's mild preference for the separate repo so it doesn't complicate their main release pipeline).

**Pinning / versioning.** The runtime package version tracks the Mojo toolchain version 1:1. When Modular releases Mojo 0.27.0, a matching `@modular/mojo-runtime@0.27.0` auto-publishes. Downstream addon authors pin the Mojo version they compiled against.

## How consumers use it

### Build-time (preferred)

Addon authors add `@modular/mojo-runtime` to their `peerDependencies` (or `dependencies`), then point their `build.sh` at it:

```bash
MOJO_RUNTIME_DIR=$(node -p "require('@modular/mojo-runtime').libDir")
mojo build --emit shared-lib --mlinker -rpath,$MOJO_RUNTIME_DIR src/lib.mojo -o build/addon.node
```

The resulting `.node` resolves its runtime deps against wherever the user's `node_modules/@modular/mojo-runtime-<platform>/lib/` ends up. No bundling inside the addon's own subpackage; no licensing ambiguity about MAX redistribution.

### Load-time (fallback, zero rebuild)

Addons that don't want to rebuild can `require('@modular/mojo-runtime').libDir` from their platform-loader `index.js` and prepend it to the appropriate env var before `require()`ing the native binary. Our [`../packages/rag/index.js`](../packages/rag/index.js) would gain ~4 lines.

### Pilot consumer

`@qkstat/rag` and `@qkstat/embed` (both in this repo) migrate first. Both have the platform-subpackage shape already; the rpath currently points into their own bundled `gpu-libs/` dir (~8 MB duplicated). After migration, that dir disappears and they depend on `@modular/mojo-runtime` instead. Binary size drops ~8 MB per platform per addon. Doubles as the proof point for Modular's announcement.

## Scope boundaries — what this is NOT

- **Not the Mojo compiler.** Users still `pixi install max` to build Mojo source. This package only ships the *runtime* libs that compiled `.node` binaries need at load time.
- **Not `stdlib.mojopkg`.** Compiled-in Mojo stdlib code is already in the user's `.node` binary. The runtime libs are a separate concern.
- **Not MAX kernels.** `linalg.mojopkg`, `layout.mojopkg`, `tensor.mojopkg`, etc. are compiled into the user's addon at build time (not loaded as shared libs). Their licensing is a separate question addressed in the main letter.
- **Not a Python wheel.** Modular already ships the runtime via `pip install modular` as part of the monolith. This proposal complements that for Node users, does not replace it.

## Ownership / handoff

| Task | Owner |
|---|---|
| Write packaging script | Wm |
| Write CI workflow | Wm |
| Draft parent/subpackage READMEs | Wm |
| Write consumer migration docs (for `@qkstat/rag` + generic) | Wm |
| Pilot `@qkstat/rag` migration | Wm |
| Provide NPM publish token for `@modular` scope | Modular |
| Review + approve scripts + CI before first publish | Modular |
| Own npm account, 2FA, deprecation policy | Modular |
| Future platform additions (linux-arm64, darwin-x64, windows-x64) | Wm writes PR, Modular merges |

**Modular's reviewer time estimate: ~2 hours** — one pass on the packaging script, one pass on CI, npm scope/token provisioning. No Mojo source changes, no MAX changes, no compiler work.

## Phased rollout

| Phase | Scope | Success metric |
|---|---|---|
| **v0.1 alpha** | linux-x64 + darwin-arm64. Single-version pin to Mojo 0.26.3. Manual publish. | `@qkstat/rag` installs + runs end-to-end on both platforms without `pixi install`. |
| **v0.2 beta** | + darwin-x64 (Intel Macs), + linux-arm64 (ARM servers). Automated publish on Mojo release tag. | Third-party Mojo-Node addon adopts it. |
| **v1.0** | + windows-x64 if/when Mojo supports Windows. Semver-stable. | Runtime package referenced in Modular's "ship a Mojo addon" docs. |

Each phase has a clean exit: if adoption stalls at v0.1, Modular can deprecate without having committed to the long-tail platforms.

## Risks & mitigations

- **glibc version targeting (Linux).** Solved the same way esbuild does — build on Ubuntu 20.04 for broad glibc compat; `libstdc++.so.6` bundled inside the package so the user's host glibc matters only for syscall layer. Existing [`../packages/rag/scripts/bundle-libs.sh`](../packages/rag/scripts/bundle-libs.sh) already does this.
- **Version drift.** Mojo runtime is still 0.x and evolving. Mitigated by 1:1 version pinning with the Mojo toolchain — `@modular/mojo-runtime@X.Y.Z` is guaranteed to match Mojo X.Y.Z.
- **Security patches.** If a CVE in libNVPTX or similar requires a hotfix, patch version bump + republish. Same flow as any esbuild patch release.
- **npm scope squatting.** Modular registers `@modular` scope immediately (if not already done) to prevent third parties from land-grabbing.
- **Licensing text in package.** Each subpackage ships a `LICENSE` file with whatever terms Modular prefers; Wm defaults to "same as MAX SDK" and Modular overrides.

## Critical files / artifacts this plan touches

**To be created (Wm writes):**

- `tools/pack-mojo-runtime-npm/` (new) — the packaging script. ~150 lines bash + Node.
- `tools/pack-mojo-runtime-npm/ci.yml` (new) — GitHub Actions or equivalent. ~60 lines.
- `docs/npm-runtime-package-consumer-guide.md` (new) — how downstream authors use it. ~80 lines.

**To be modified (Wm writes PRs after packages publish):**

- [`../packages/rag/build.sh`](../packages/rag/build.sh) — replace pixi-lib lookup with `@modular/mojo-runtime` resolution.
- [`../packages/rag/scripts/bundle-libs.sh`](../packages/rag/scripts/bundle-libs.sh) — deprecate (no longer needed).
- [`../packages/rag/index.js`](../packages/rag/index.js) — add `@modular/mojo-runtime` resolution to set `DYLD_LIBRARY_PATH` before loading.
- [`../packages/rag/package.json`](../packages/rag/package.json) — add `@modular/mojo-runtime` as dep, remove bundled-libs file from `files`.
- [`../packages/embed/`](../packages/embed/) — parallel migration.

**Reused as templates, not modified:**

- [`../packages/rag/npm/linux-x64/package.json`](../packages/rag/npm/linux-x64/package.json) — `os`/`cpu` shape.
- [`../packages/rag/index.js`](../packages/rag/index.js) lines 9–46 — platform resolver pattern.

## Verification — how we know it works

**End-to-end test script** (ships with the proposal, runnable by Modular reviewers):

1. Fresh Ubuntu 22.04 container, no pixi, no MAX, no Mojo.
2. `npm install @qkstat/rag @modular/mojo-runtime`.
3. `node -e "const r = require('@qkstat/rag'); console.log(r.search(new Float32Array(384), 10))"`.
4. Expected: semantic search runs, returns results, zero references to a non-existent `.pixi/` or `/opt/modular/` dir.

**Compatibility regression test** — the existing [`../spikes/mojo-runtime/`](../spikes/mojo-runtime/) tier-0..4 experiment becomes a CI job. On every Mojo version bump, rebuild the five tiers against `@modular/mojo-runtime@<new-version>` and verify `ldd` output matches the pinned manifest in the subpackage. Catches any unannounced runtime-lib additions.

**Smoke test at publish time** — `npm publish --dry-run` on all three packages; verify they install cleanly and `require()` resolves in a scratch Node project. Runs in CI before the real publish step.

## Open questions for Modular (surface before kickoff)

1. **npm scope.** Is `@modular` already registered on npmjs.com? If not, register before anything else. If registered but used for another purpose, pick an alternate (e.g., `@modular-sh`).
2. **CI home.** New repo (`modular/mojo-runtime-npm`) vs workflow in `modular/modular`? Wm's mild preference for separate repo so it doesn't complicate their main release pipeline.
3. **License field in published packages.** Same as MAX SDK? Apache-2.0 with attribution? Need Modular's call.
4. **NOTICE file contents.** What attribution text does Modular want shipped with each subpackage?
5. **Release cadence alignment.** Do we publish every nightly, every stable, or on-demand? My recommendation: stable-only for v0.1, then evaluate.

Once answered, Wm can start the packaging-script PR the same day.

## Timeline

- **Week 1:** Packaging script + Linux subpackage. Publishes to a `@wmtalcott` scope for testing.
- **Week 2:** Darwin subpackage. Consumer migration in `@qkstat/rag`. End-to-end test passes.
- **Week 3:** CI pipeline hooked up. Dry-runs only.
- **Week 4:** Modular review pass. First real publish under `@modular` scope. `@qkstat/rag` production release depending on it.

Total: **~1 engineering week of work** spread over 4 calendar weeks to accommodate Modular review cycles.
