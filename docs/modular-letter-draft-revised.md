Hi Modular licensing team,

## What we're shipping

`@qkstat/rag` — an npm package for Node.js. Inside it is `gpu.node`, a
Node.js N-API shared library we build from Mojo source using
`mojo build --emit shared-lib`. It contains one cached GPU matmul kernel
(`linalg.matmul[target="gpu"]`) and a top-k retrieval primitive for RAG
workloads. Our initial prebuild matrix:

* Linux x86_64 + NVIDIA — `--target-accelerator sm_80`
    (PTX forward-compat via driver JIT on sm_80/86/89/90/100+)
* Darwin arm64 + Apple Silicon — `--target-accelerator metal:4`
* AMD / additional targets to follow on the same source.

To deliver a clean `npm install` (no separate `pixi install max` required),
we want to bundle runtime libraries inside the platform-specific subpackage
(`@qkstat/rag-linux-x64`, `@qkstat/rag-darwin-arm64`) with rpath set to
`$ORIGIN/gpu-libs`, so only our addon resolves them.

## What the compiled binary actually links against

We built five small Mojo shared libraries, progressively stripping imports
to isolate which dynamic dependencies come from which feature. Source,
methodology, and full `ldd` / `otool` captures at:
<https://github.com/codetalcott/mojo-addon-examples/tree/spike/mojo-runtime-isolation/spikes/mojo-runtime>
(writeup:
<https://github.com/codetalcott/mojo-addon-examples/blob/spike/mojo-runtime-isolation/docs/mojo-runtime-isolation-spike-findings.md>).

| Tier | Imports beyond prelude | GPU target |
|---|---|---|
| 0 | `std.math` only | none |
| 1 | + `std.algorithm` (vectorize/parallelize), `std.memory` | none |
| 2 | + `std.gpu`, `std.gpu.host` — hand-rolled kernel via `UnsafePointer` | yes |
| 3 | + `layout` package (TileTensor) | yes |
| 4 | + `linalg.matmul[target="gpu"]` — our `packages/rag` baseline | yes |

On H100 Linux (Mojo `0.26.3.0.dev2026041716`, `ldd` direct deps), tier 1 —
a CPU-only binary with no GPU code, no `--target-accelerator`, no
`std.gpu`, no `layout`, no `linalg` — links:

```
libKGENCompilerRTShared.so
libAsyncRTMojoBindings.so
libAsyncRTRuntimeGlobals.so
libMSupportGlobals.so
libNVPTX.so
libstdc++.so.6
libgcc_s.so.1
+ glibc (libc, libm, libdl, ld-linux, linux-vdso)
```

Tiers 2, 3, 4 add **zero** new dynamic libraries on top of tier 1. The
full set above is emitted by the Mojo compiler for any binary that uses
the async runtime or allocates memory — including `libNVPTX.so`, which is
present in the CPU-only tier 1 binary with no GPU code path (so it's not
GPU-conditional).

`linalg.matmul` *does* add ~98 KB of statically-embedded PTX kernel text
(`linalg::gemv::gemv_split_k` symbols per `nm` on the Linux sm_80 binary),
so the **compiled-in kernel code is a separate question from the dynamic
library question** — we want to address both cleanly below.

## The questions

Compiled Mojo binaries cannot function without the libraries listed above.
A binary using only `std.mojopkg`'s async runtime or allocator pulls them
all in. Since `std.mojopkg` is released under Apache 2.0 with LLVM
exceptions (per Modular's March 2024 open-source announcement), these
libraries function as the required execution environment for anything
compiled from the open-source standard library — structurally analogous
to libgcc and libstdc++ under the GCC runtime exception.

**Q1. Are the Mojo compiler runtime libraries (libKGENCompilerRTShared,
libAsyncRTMojoBindings, libAsyncRTRuntimeGlobals, libMSupportGlobals,
libMGPRT, libNVPTX, and any future target-specific equivalents) considered
Redistributable Components under the current MCL's standalone-redistribution
clause, or are they implicitly redistributable as the Mojo execution
environment — the GCC-runtime-exception analog for Mojo?**

We'd like to bundle them at `$ORIGIN/gpu-libs` inside our npm platform
subpackages so our users can `npm install @qkstat/rag` without a separate
MAX install step. They'd be reachable only through our `gpu.node`'s rpath —
not callable by other applications without explicit effort.

**Q2. Does compiled output of `linalg.mojopkg` (and `layout.mojopkg`)
embedded in our binary via `linalg.matmul[target="gpu"]` fall under the
Redistributable Components definition?**

Neither `linalg` nor `layout` is in `stdlib.mojopkg`; Modular forum
thread #662 (Joe L.) confirms `tensor`, `kv_cache`, `register`, `max`
are explicitly "not stdlib" but the status of `linalg`/`layout` is not
publicly documented. If the compiled kernel output is MCL-covered and
redistribution of such compiled output isn't permitted as part of an
Application, we can refactor to a hand-rolled kernel using only
`std.gpu` (proven buildable at tier 2 — see the spike repo). That's a
performance hit on H100 we'd rather avoid (`linalg` dispatches to TF32
tensor cores; a hand-rolled FP32 kernel without tensor-core intrinsics
would be meaningfully slower), but we'd rather do the work than ship
something whose license status is uncertain.

**Q3. If both are permitted, what are the attribution / notice
requirements?** A NOTICE file in the npm package citing MAX / Mojo
versions bundled? Specific wording you'd like us to use? Our project is
non-commercial at launch; production deployments fall under the free
NVIDIA / CPU tier per the January 2026 licensing update.

## A counter-proposal worth mentioning

If Modular would consider publishing the Mojo compiler runtime as a
separately-installable platform-specific package (npm, conda, pip —
equivalent to how `@modular/max-runtime-linux-x64` would work), that'd
resolve the redistribution question for our case and anyone else
shipping Mojo-compiled Node / Python addons: authors depend on
Modular's package, Modular handles redistribution directly. Happy to
pilot that pattern if it's useful.

## Project context

* Source (open): github.com/codetalcott/mojo-addon-examples
    (currently private pending this conversation; going public on
    resolution)
* Audience: Node.js developers doing exact-retrieval RAG on
    GPU-accelerated hosts (initial prebuilts for NVIDIA + Apple;
    AMD to follow)
* Benchmarks to date: sub-100 µs exact semantic search at recall=1.0
    on H100 (sm_90)
* Ecosystem involvement: we maintain napi-mojo
    (github.com/codetalcott/napi-mojo), an open-source N-API binding
    framework for Mojo — ~140 exported functions, ~620 Jest tests —
    published on npm since early 2026. `@qkstat/rag` is built on top
    of napi-mojo's N-API patterns.

Thanks,
Wm Talcott
