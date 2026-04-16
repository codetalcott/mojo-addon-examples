# Spike Findings Log

Living document — updated as each gate resolves. Separate from
[ideas/embedding-kernel-spike-plan.md](../../../ideas/embedding-kernel-spike-plan.md)
(the plan, mostly static) and
[ideas/killer-kernel-and-agent-cli.md](../../../ideas/killer-kernel-and-agent-cli.md)
(the thesis, also static).

Captures what the spike actually discovered, which often differs from what the
plan assumed. Anyone reviving this spike in six months should read this first.

---

## Day 0 — infrastructure

- **Lambda Cloud had no H100 capacity** when the spike started. Switched to RunPod on 2026-04-16. $300 Lambda credit sunk; kept `scripts/lambda-bench.sh` in the tree in case capacity returns.
- **RunPod Secure Cloud H100 SXM is ~$2.99/hr** in US-NE-1, US-MO-1, US-CA-2 (as of 2026-04-16). "Low" stock reported by the API sometimes means zero — capacity probe with a real launch was the only reliable signal.
- **Mojo-addon-examples repo structure now assumes pods** — `scripts/bootstrap.sh` is the canonical pod-side session setup; seeded onto RunPod Network Volumes at `/workspace/persist/bootstrap.sh`.
- **Pod cold-start ~30s** after pixi env is cached on the Network Volume. First cold build of `packages/rag` + `spike` is ~30s with cache, ~5 min without.

## Day 1 — Gate F1 (model loadability)

### What we thought we'd find

Per the original plan: write `spike/src/embed.mojo` with `from max.graph import Graph, ops` etc., load MiniLM-L6-v2 via a Mojo-native API, run forward on `DeviceContext`.

### What we actually found

**There is no Mojo-native model loader in MAX v26.** The high-level inference surface is Python-only:
- `max.graph.Graph`, `max.nn.Module`, `max.engine.InferenceSession` — Python
- `max/c/model.h` C API exists (`M_compileModel`, `M_executeModelSync`) but expects pre-compiled graphs from Python tooling
- Compiled `.mojopkg` files under `.pixi/envs/default/lib/mojo/` (`nn`, `tensor`, `weights_registry`, `pipeline`) expose primitive ops + scheduling — not model loaders
- Zero Mojo code in the portfolio actually calls `max.graph` or `max.engine`; one abandoned attempt in `onedev-mojo-agent/src/mojo_max/` tried FFI to non-existent symbols

The sanctioned path to use MAX from Mojo: **Python interop** via `from std.python import Python`. Documented in `skills/mojo-python-interop/SKILL.md`. The `mojo-addon-examples` repo has zero examples of this yet.

### Gate F1 verdict: PASS (via Python interop)

Python-interop F1 validation script: [`spike/probe_dlpack.py`](probe_dlpack.py). Ran on H100 (`sm_90a`):

- `driver.Accelerator()` → H100 device recognized (`api=cuda`, `arch=sm_90a`)
- `Graph(...)` with `DeviceRef.GPU()` compiles
- `InferenceSession.load(graph)` returns a compiled `Model`
- `model.execute(Buffer.from_numpy(x).to(gpu))` runs on GPU
- Output is `max.driver.Buffer` on `Device(type=gpu, id=0)`, `is_host=False`
- `r0.__dlpack__()` returns PyCapsule `"dltensor"` (`DLManagedTensor*`)
- `r0.__dlpack_device__()` returns `(2, 0)` = DLPack `kDLCUDA`
- `Buffer.from_dlpack(r0)` round-trips cleanly, stays on GPU

### M4 is not a viable local fallback

**`Accelerator()` on Apple M4 raises** `"Not implemented for device: Apple M4"`. MAX CPU path works locally (Python-level graph construction + CPU execute OK), but any GPU work must happen on RunPod.

Impact on plan: the "M4 for correctness, RunPod for bench" split from the original plan gets revised. **All GPU correctness iteration happens on RunPod.** Cost stays manageable because pod launches are ~$0.15 per round-trip, and $300 remaining in the budget covers 40+ hours.

## Day 1 — MAX v26 API cheat sheet

API differs from what pretrained model training data suggests. Use these:

| What you want | v26 syntax |
|---|---|
| GPU device handle | `driver.Accelerator()` (NOT `driver.GPU()`) |
| Graph device spec | `DeviceRef.GPU()` in `TensorType` |
| Host→GPU transfer | `driver.Buffer.from_numpy(x).to(dev)` |
| Compile | `session = InferenceSession(devices=[dev]); model = session.load(graph)` |
| Execute | `model.execute(buf1, buf2, ...)` — inputs must be on target device |
| GPU output | List of `driver.Buffer`, `__dlpack__()` + `__dlpack_device__()` work |
| Load weights | `session.load(graph, weights_registry={"layer.weight": tensor, ...})` |
| Python-in-Mojo import | `var max_engine = Python.import_module("max.engine")` |

Confirmed working combos stored in [`probe_dlpack.py`](probe_dlpack.py) captures under `docs/runpod-dlpack-probe-*.txt`.

## Day 1 — budget-relevant decisions for Day 2+

1. **Python interop is the Mojo→MAX bridge.** Spike stays in the main `@qkstat/rag` shape (Node N-API addon built with napi-mojo); Mojo code calls `Python.import_module("max.engine")` for model loading and execution.
2. **Zero-copy handoff exists** via DLPack (`__dlpack__()`) — the PyCapsule contains a GPU pointer we could wrap as `DeviceBuffer(ctx, ptr, count, owning=False)` from Mojo. **Deferred to a follow-up optimization** — D2H/H2D bounce is ~20μs at d=384, negligible vs. ~ms embedding compute.
3. **Unknown: does Mojo `DeviceContext()` share CUDA context with MAX's?** If not, zero-copy needs care (or we bounce through host). 1–2 hour verification task when we implement the zero-copy path.
4. **Building MiniLM-L6-v2 by hand in `max.nn` is non-trivial** (~4-6 hours of Python + weight adapter). For Day 2 we ship a pipeline skeleton with a trivial graph; Day 3 swaps in real MiniLM.

---

## Day 2 — to be filled in

(embedding pipeline skeleton)

## Day 3 — to be filled in

(real MiniLM-L6-v2 with weight adapter)

## Day 5 — to be filled in

(benchmark — Gate F2 / F3)

## Day 10 — to be filled in

(full demo + writeup + GO/NO-GO)
