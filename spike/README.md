# spike/ — embedding-kernel spike (COMPLETE)

Isolated workspace for the 2-week embedding-kernel spike. Completed
2026-04-17 in 2 elapsed days, ~$10 of RunPod compute.

- **Plan**: [`../../ideas/embedding-kernel-spike-plan.md`](../../../ideas/embedding-kernel-spike-plan.md)
- **Day-by-day execution log**: [`findings.md`](findings.md)
- **Decision artifact / writeup**: [`../../ideas/embedding-kernel-spike-writeup.md`](../../../ideas/embedding-kernel-spike-writeup.md)

## Verdict

**GO** on the killer-kernel portfolio path. All five gates passed:

| Gate | Target | Measured |
|---|---|---|
| F1 — Loadability | MiniLM-L6-v2 runs on H100 | ✓ `InferenceSession.load` on `sm_90a` |
| F2 — Latency (batch-1) | ≤ 5 ms p50 | **1.44 ms** (seq-32) |
| F2 — Latency (batch-64) | ≤ 20 ms p50 | **17.88 ms** (seq-128, 1k corpus) |
| F3 — Tokenize overhead | ≤ 20 ms | **0.06 ms** (batch-1 seq-32) |
| F4 — Correctness | cosine ≥ 0.995 vs CPU reference | **0.99999 min**, 0.999998 mean |

End-to-end query total (tokenize + embed + search + top-10) on H100:
**1.44 ms p50 at 1k-doc corpus, 1.81 ms at 10k**.
**~2× faster than** `@huggingface/transformers` + `hnswlib-node` CPU stack,
**~130× faster** for bulk corpus indexing.

## Layout

```text
spike/
├── src/
│   ├── lib.mojo                     # N-API entry point (register_module)
│   └── embed.mojo                   # embed_tokens_fn — Python interop → MAX
├── build.sh                         # Compiles src/lib.mojo → build/embed.node
├── package.json                     # @huggingface/transformers + hnswlib-node
│
├── embed.py                         # Python-side: load MAX graph + embed_batch_from_addrs()
├── bert_graph.py                    # Vendored (Apache-2.0) — BERT graph + BertModelConfig
├── bert_weight_adapter.py           # Vendored — HF → MAX weight name map
├── tokenize.js                      # WordPiece via @huggingface/transformers v4
├── corpus.js                        # Programmatic clustered corpus (shared by demo + bench)
│
├── reference.js                     # CPU ground-truth generator (ONNX)
├── probe_dlpack.py                  # Day 1 DLPack zero-copy probe
├── test-roundtrip.js                # Gate F4 — cosine vs CPU reference
├── demo.js                          # Day 4 — GPU end-to-end with packages/rag
├── demo-reference.js                # Day 4 — CPU ONNX + hnswlib comparison
├── bench.js                         # Day 5 — warm-path per-shape benchmarks
│
├── findings.md                      # Execution log (Day 0–5)
├── fixtures/                        # ground-truth.bin, sanity-set.txt (git-ignored)
└── build/                           # embed.node (git-ignored)
```

## Running from a clean RunPod pod

Requires `scripts/runpod-launch.sh` setup
(see [plan Day 0](../../../ideas/embedding-kernel-spike-plan.md) and
[`scripts/bootstrap.sh`](../scripts/bootstrap.sh)).

```bash
source ~/.config/runpod/env

# Gate F4 — correctness
./scripts/runpod-launch.sh -- \
  "cd /workspace/mojo-addon-examples && git fetch && git reset --hard origin/spike/embedding-kernel && \
   pixi run bash spike/build.sh && pixi run node spike/test-roundtrip.js"

# Day 4 demo — spike vs reference head-to-head
./scripts/runpod-launch.sh -- \
  "cd /workspace/mojo-addon-examples && git fetch && git reset --hard origin/spike/embedding-kernel && \
   pixi run bash packages/rag/build.sh && pixi run bash spike/build.sh && \
   pixi run node spike/demo.js && pixi run node spike/demo-reference.js"

# Day 5 bench — warm-path per-shape latencies
./scripts/runpod-launch.sh --max-runtime-hours 2 -- \
  "cd /workspace/mojo-addon-examples && git fetch && git reset --hard origin/spike/embedding-kernel && \
   pixi run bash packages/rag/build.sh && pixi run bash spike/build.sh && \
   pixi run node spike/bench.js"
```

Each run is ~2-4 min on H100, ~$0.15-0.30.

## Fate of this directory

On **GO** (current verdict): primitives get promoted out of `spike/` into a
sibling package of `packages/rag/` (`packages/embed/` proposed). This
directory stays as the reference implementation until the promotion lands.

On **NO-GO**: keep `spike/` as-is for the next person who wants to try —
the scaffold is the useful residue.

## What's inherited when you move this to `packages/embed/`

- The Mojo N-API binding ([`src/embed.mojo`](src/embed.mojo))
  — clean, compiles against current napi-mojo + MAX 26.3+
- The Python-side engine ([`embed.py`](embed.py)) — module-cached singleton,
  `embed_batch_from_addrs` is the N-API-friendly entry point
- The vendored BERT graph ([`bert_graph.py`](bert_graph.py)) — Apache-2.0,
  stays until we either contribute a lightweight upstream entry point or
  MAX ships a `max.sentence_transformers` convenience API
- The tokenize → embed → dst-memmove zero-copy-ish path (raw-address Python
  interop) — fast enough for 1.4 ms per query, optimization surface is in
  DLPack zero-copy if sub-1 ms is ever required (Phase 2)

## What's NOT inherited

- **M4 Metal support** — MAX reports "Not implemented for device: Apple M4"
  at `Accelerator()` init. All GPU correctness iteration required RunPod.
  Track MAX releases for when this lands.
- **Async N-API** — spike is sync-only. Phase 1a of the
  [rag-productization plan](../../../ideas/rag-productization.md) addresses
  this across both the embed and rag kernels.
- **Prebuilt distribution** — gated on the Modular license question from
  `packages/rag/npm/`. The spike validates the code; the packaging story
  is orthogonal.

## Known caveats

- **29.6 s CUDA kernel JIT cold-start** on first forward. One-time per
  process. Irrelevant for long-running services; painful for CLI tools.
  AOT-compiling the kernels via MAX's offline toolchain is a
  2-day follow-up investigation.
- **Batch-64 search is JS-level serial.** `rag.node`'s `searchHandle`
  supports multi-row, but `GpuIndex.search` calls it per-query in a loop.
  Would benefit from `GpuIndex.searchBatch` — easy optimization.
- **Cluster-accuracy numbers are on synthetic data.** Production recall
  needs MS-MARCO; the [`examples/matmul/matmul_rag.js`](../examples/matmul/matmul_rag.js)
  fixture scheme in the parent repo is ready to adopt.
