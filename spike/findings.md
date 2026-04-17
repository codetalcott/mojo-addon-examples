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

## Day 2 — Python embedding engine

### What works now

[`spike/embed.py`](embed.py) runs `sentence-transformers/all-MiniLM-L6-v2` on H100 via MAX:

```text
embeddings shape: (2, 384)  dtype: float32
latency: 9.8ms for batch-2   (cold-start, includes H2D + D2H)
```

Capture: `docs/runpod-embed-day2-20260416T232817Z.txt`.

### Key discoveries that shaped Day 2

1. **MAX ships a BERT pipeline for MiniLM-L6-v2.** [Upstream source](https://github.com/modular/modular/tree/main/max/python/max/pipelines/architectures/bert). `BertModel` lists `sentence-transformers/all-MiniLM-L6-v2` and `all-MiniLM-L12-v2` as supported. No need to hand-build the transformer graph.
2. **The `max.pipelines` framework has heavy deps** — pydantic, pillow, msgspec, and a growing list for diffusion/vision/LLM models. For our embedding-only spike, we vendored the two files we actually need (`graph.py`, `weight_adapters.py`) as [`spike/bert_graph.py`](bert_graph.py) + [`spike/bert_weight_adapter.py`](bert_weight_adapter.py), with a minimal duck-typed `BertModelConfig`. Saves ~300MB of deps and avoids churn as upstream's architecture registry evolves.
3. **`safetensors.numpy` is the right loader.** `safetensors.torch` requires PyTorch. `safetensors.numpy.load_file` returns `dict[str, np.ndarray]` directly — zero dep on torch.
4. **`transformers`'s "PyTorch was not found" warning is cosmetic.** We only use `AutoConfig.from_pretrained()` for BERT architecture params — no model loading through transformers.
5. **L2-normalization isn't automatic.** `BertModel` with `pool_embeddings=True` does mean-pooling but not final L2-normalize. sentence-transformers adds this. Spike will add L2-normalize either in `embed.py` or in Mojo post-processing.

### Lockfile + bootstrap hygiene

Committing `pixi.lock` caused constant conflicts on pods (pixi regenerates it on every `pixi install`). Removed from tracking and gitignored.

Pod-side `bootstrap.sh` needed a `git reset --hard` before the checkout to discard local modifications. Fixed in `scripts/bootstrap.sh`. Volume bootstrap was updated on 2026-04-16.

`scripts/runpod-launch.sh` now soft-sources bootstrap (failure doesn't abort the wrapper) and the command itself is expected to sync the repo state. This pattern is more robust to stale volumes.

### Deps added

```toml
[dependencies]
# Python spike side
pydantic = ">=2.0"          # required by huggingface_hub
transformers = ">=4.40"     # AutoConfig for BERT architecture params
safetensors = ">=0.4"       # numpy loader for model weights
huggingface_hub = ">=0.20"  # snapshot_download for MiniLM assets
pillow = ">=10.0"           # transitive, max.interfaces imports PIL
```

~120MB added to the pixi env. All conda-forge.

### Still pending (Day 3)

1. **Mojo-side integration.** Rewrite [`spike/src/embed.mojo`](src/embed.mojo) `embed_tokens_fn` to:
   - `Python.add_to_path("/workspace/mojo-addon-examples/spike")`
   - `Python.import_module("embed")`
   - Create / cache an `EmbeddingEngine` instance
   - Call `engine.embed_batch(ids_np, mask_np)` and write result to N-API dst buffer
2. **L2-normalize.** Either in `embed.py` (add `embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)`) or in Mojo.
3. **Correctness gate (F4).** Compare Mojo-produced embeddings to `spike/reference.js` output; expect cosine similarity ≥ 0.995. Ideally run the sanity set through both pipelines and diff.
4. **Latency probe.** 9.8ms for cold batch-2 isn't definitive. Need warm single-query + warm batch-64 numbers. That's Day 5 (Gate F2).

### Numbers to beat on Day 5

From the spike plan:

- Batch-1: ≤ 5ms p50
- Batch-64: ≤ 20ms p50

Today's 9.8ms batch-2 cold-start is preliminary; warm batch-1 likely ≥ 3ms (dominated by kernel launch overhead at this shape). TBD.

## Day 3 — Mojo↔Python bridge + Gate F4 PASS

### What works now (end-to-end, on H100)

```text
min cosine:  0.999990   (target ≥ 0.995)
mean cosine: 0.999998
cold:  27.9 s  (model download + MAX graph compile + first kernel launch)
warm:  2.82 ms/op  batch-100 × seq_len-14  (20-iter average)
```

Full pipeline: JS tokenize → N-API `embedTokens` → Mojo `embed_tokens_fn`
→ `Python.import_module("embed")` → `embed.embed_batch_from_addrs` (Python)
→ `EmbeddingEngine.embed_batch_l2` → MAX graph on H100 → fp32 embeddings
→ ctypes.memmove into JS-owned `Float32Array`.

Capture: `docs/runpod-day3-20260416T234521Z.txt`.

### Numerical parity to CPU reference

Comparison is against `@huggingface/transformers` CPU-side (Xenova's ONNX
MiniLM) on the same 100-sentence sanity set. **6-decimal-place agreement**
across all 100 sentences. MAX's fp32 BERT forward matches ONNX Runtime's
fp32 BERT forward essentially to the bit-representation precision.

Worst offender: sentence 36 ("Comments explain why, not what.") at cosine
0.999990. The minor drift is in the last 3 float32 decimal places —
dominated by ULP-level accumulation order differences, not model bugs.

### Mojo ↔ Python handoff details

- Mojo gets raw pointers from N-API (`JsTypedArray.data_ptr`).
- `Int(ptr)` implicitly converts the pointer to integer address.
- Python reads via `numpy.ctypeslib.as_array((c_int32 * n).from_address(addr))`
  — zero-copy view of the JS memory.
- Output: `ctypes.memmove(dst_addr, result.ctypes.data_as(c_void_p), nbytes)`
  writes into the JS Float32Array directly.
- `EmbeddingEngine` instance is a module-level singleton in `spike/embed.py`;
  first N-API call pays the 28s cold-start, subsequent calls reuse the
  compiled graph.

### Gotchas found during Day 3

1. **`@huggingface/transformers` v4 changed the tokenizer return API.** v2
   (Xenova) returned nested JS arrays for `input_ids`; v4 returns a Tensor
   object with `.dims = [batch, seqLen]` and `.data = BigInt64Array`. Had to
   rewrite `spike/tokenize.js` to use `.dims` + down-cast BigInt → Number.
2. **`throw_js_error` takes `StringLiteral`, not `String`.** Mojo error
   formatting can't be used with this API. The Python exception details go
   to stderr instead; good enough for a spike.
3. **`pixi.lock` and `package-lock.json` churn on the pod.** Now
   gitignored; bootstrap does `git reset --hard HEAD` before checkout.
4. **`spike/fixtures/` directory needs `mkdir -p`** before first write —
   .gitignore'd contents means the dir doesn't exist on a fresh clone.
   `spike/reference.js` now creates it.

### Gate F2 — preliminary PASS

Spike plan targets for batch-64: ≤ 20 ms p50. Today's batch-100 × seq_len-14
at **2.82 ms** crushes that by ~7×. Caveats:

- Real query shape is batch-1, seq_len ~ 16–128. Per-op fixed overhead
  (Python FFI + MAX kernel launch) is likely 1–3 ms regardless of batch.
  So batch-1 will look WORSE per-sample than batch-100.
- Longer sequences (seq_len=128 for production MS-MARCO passages) will
  increase compute cost ~10× compared to seq_len=14.
- These are warm-path numbers after ~20 iterations. Cold-start is 28s
  (acceptable for a long-running service; fatal for scripts/CLI).

Day 5 benchmarks will measure batch-1/8/32/64 at seq_len=32/128 to give
honest per-shape numbers.

### What's left for Day 4

1. Wire a proper demo: `spike/demo.js` with a small corpus (say 1000
   MS-MARCO passages), tokenize + embed + search via `packages/rag`
   primitives. End-to-end "local GPU semantic search from Node."
2. Matmul + top-k path — the `packages/rag` kernels already do this;
   need to make sure the embeddings we produce feed into them cleanly.
3. `spike/demo-reference.js` — same flow using `@huggingface/transformers` and
   `hnswlib-node` (no MAX, no Mojo) for apples-to-apples comparison.

### Numbers that now matter for Day 10

- **Warm batch-1 at seq_len=32**: expected ~1-2 ms based on Day 3 data
- **Warm batch-1 at seq_len=128**: expected ~2-5 ms
- **End-to-end (tokenize + embed + search + top-10) for a 10k corpus**:
  target ≤ 5 ms on H100 per query

## Day 4 — end-to-end demo

### What works

Two addons in one Node process, each with its own CUDA context, cleanly
composed:

1. `spike/build/embed.node` — MiniLM-L6-v2 forward via MAX on H100
2. `packages/rag/build/rag.node` — exact cosine search + top-k via `linalg.matmul`

Host-bounce between them (numpy → JS Float32Array → `loadMatrixGpu`) adds
~40μs per query. Negligible vs. ~1.6ms embed + 0.09ms search.

### Demo numbers (1000-doc programmatic clustered corpus, k=10)

```text
Spike (MAX GPU + packages/rag exact):
  per-query (warm):
    embed   p50 = 1.68 ms   avg = 1.58 ms
    search  p50 = 0.086 ms  avg = 0.088 ms
    total   p50 = 1.76 ms   avg = 1.67 ms
  accuracy: 100/100 in-cluster hits

Reference (ONNX CPU + hnswlib-node):
  per-query (ef=100, warm):
    embed   p50 = 2.91 ms
    search  p50 = 0.18 ms
    total   p50 = 3.13 ms
  accuracy: 100/100 in-cluster hits
```

Capture: `docs/runpod-day4-20260416T235942Z.txt`.

**Spike is 1.8× faster end-to-end for single-query retrieval on a 1k corpus.**
The gap will widen with larger corpora (search is O(N) on our exact path but
GPU-parallel; HNSW is O(log N) but CPU-serial).

### Demo surprise: corpus embed looks slow but isn't

`spike/demo.js` reports "corpus embed: 33.75s (30 docs/sec)" while reference
runs "7.0s (143 docs/sec)". Looks like the GPU path is 5× slower for bulk,
which would contradict the headline.

**It's almost entirely CUDA kernel JIT on the first batch.** Breakdown:

- Cold first `embedTokens` call: ~33s (model load cached on volume + CUDA
  kernel compile + first forward)
- Subsequent 15 warm batches: ~30ms total (~2ms/batch-64)

For a long-running service (API, daemon, scheduled job) the cold cost is
paid once and becomes irrelevant. For scripts or CLI tools the ~30s cold
start matters. Day 5 benchmarks will separate cold vs. warm explicitly.

**Reference doesn't have this shape** because ONNX Runtime's CPU EP JIT is
per-op and per-CPU-feature, typically ~seconds total rather than per-kernel
on GPU. Not a property of the model architecture, a property of how each
runtime initializes.

### Incidental fix landed: rag against MAX 26.3+

`packages/rag/src/kernels.mojo` stopped compiling because MAX's
`HostBuffer.unsafe_ptr()` now returns `Optional[UnsafePointer[...]]`
instead of a plain pointer. Fixed by calling `.value()` before
`.bitcast[Byte]()`. Unrelated to the spike's thesis, but the spike caught
the regression because Day 4 needed `rag.node` built alongside `embed.node`.

### What's left for Day 5 (benchmarks)

1. **Separate cold vs. warm latency** — right now corpus embed time is
   dominated by first-batch JIT. Need to report p50 over warm batches only.
2. **Per-shape latency** — batch-1, batch-8, batch-64 at seq_len-32 and
   seq_len-128 (production-representative).
3. **Separate timers** — tokenize, H2D, forward, D2H, matmul, top-k.
4. **Larger corpus** — 10k and 100k docs to see where exact cosine matmul
   breaks even with HNSW's log(N) advantage.
5. **Throughput** — QPS at sustained batch-1 (what a serving API looks like).

## Day 5 — benchmarks (Gates F2, F3 PASS)

### Warm-path numbers on H100 (post-JIT, 50-iter p50)

```text
CORPUS 1000 docs
  batch-1  seq-32    total=1.44 ms  (p95=2.13, p99=2.30)
  batch-1  seq-128   total=2.68 ms  (p95=3.17, p99=3.34)
  batch-64 seq-32    total=8.64 ms
  batch-64 seq-128   total=17.88 ms

CORPUS 10000 docs
  batch-1  seq-32    total=1.81 ms
  batch-1  seq-128   total=2.89 ms
  batch-64 seq-128   total=18.67 ms

Corpus-embed warm throughput: 18,448 docs/sec (batch-64, post-JIT)
Cold-start (one-time): 29.6 s
```

Capture: `docs/runpod-day5-bench-20260417T001138Z.txt`.

### Per-stage timing (batch-1 seq-32, 1k corpus)

| Stage | p50 |
|---|---|
| Tokenize (JS, `@huggingface/transformers`) | 0.06 ms |
| Embed (MAX GPU forward + D2H) | 1.30 ms |
| Search (`packages/rag` matmul + top-k) | 0.084 ms |
| **Total** | **1.44 ms** |

Tokenize is 4% of total latency — not a bottleneck. F3's 20 ms budget had
300× headroom.

### Gate F2 / F3 verdicts

- **F2 batch-1**: target ≤ 5 ms, measured **1.44 ms seq-32 / 2.68 ms seq-128** → **PASS** (3× headroom)
- **F2 batch-64**: target ≤ 20 ms, measured **8.64 ms seq-32 / 17.88 ms seq-128** → **PASS**
- **F3 tokenize**: target ≤ 20 ms, measured **0.06 ms / 0.44 ms** → **PASS** (300× headroom)

### Observations

1. **Cold-start is CUDA kernel JIT.** 29.6 s on first batch; subsequent batches are warm. Day 4's "corpus embed 33 s" confusion explained — it was almost entirely first-batch JIT.
2. **Batch-64 search is JS-level-serial.** `rag.node`'s `searchHandle` supports multi-row input natively, but `GpuIndex.search` wraps it as a JS loop. Fixing this (→ `GpuIndex.searchBatch`) would drop batch-64 search from ~3 ms to sub-ms.
3. **Search grows with N.** 0.084 ms on 1k vs 0.158 ms on 10k at batch-1 — that's the O(N) matmul. Still tiny in absolute terms; at 100k corpus expect ~1-2 ms per query total.
4. **HNSW vs exact crossover is favorable.** Reference's CPU ONNX + hnswlib ef=100 is 3.13 ms total at 1k. Spike is 1.44 ms at 1k, 1.81 ms at 10k. Spike stays ahead as corpus grows (O(N) matmul scales better than expected given H100's memory bandwidth).

### Compared to CPU reference

| | Spike (MAX H100) | Reference (ONNX CPU + hnswlib ef=100) |
|---|---|---|
| Per-query total (batch-1 seq-32, 1k corpus) | 1.44 ms | 3.13 ms |
| Corpus embed throughput (warm) | 18,448 docs/sec | 143 docs/sec |
| Recall | 1.0 (exact) | 1.0 (on easy clustered corpus) |
| Cold-start | 29.6 s | 1.1 s |

**~2× faster per query, ~130× faster for corpus indexing** at honest warm numbers.

---

## Day 10 — writeup complete

See [`ideas/embedding-kernel-spike-writeup.md`](../../../ideas/embedding-kernel-spike-writeup.md)
for the portfolio-level GO/NO-GO artifact. Spike ends here; next steps are
in Phase 1 of the productization plan.

**Verdict: GO** on the killer-kernel path.

**For the tactical 2–4 week plan following this spike**, see
[`ideas/post-spike-next-steps.md`](../../../ideas/post-spike-next-steps.md)
— critical path (promote `spike/` → `packages/embed/`, wire async N-API,
resolve Modular license, MS-MARCO bench, publish writeup) plus parallel
investigations and decision gates. The strategic 10-week plan in
[`ideas/rag-productization.md`](../../../ideas/rag-productization.md) stays
canonical; Phase 3b (embedding) is now pulled forward to Phase 1b.

### Budget actuals

- Pod-hours: ~3 H100 SXM at $2.99/hr
- Total spend: **~$10**
- Laptop iteration time: ~8 hours focused
- Days elapsed: 2 (planned: 10 working days)

Well under the budget ceiling the spike plan anticipated.

## Day 10 — to be filled in

(full demo + writeup + GO/NO-GO)
