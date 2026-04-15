# Sub-100 µs exact semantic search from Node, with recall = 1.0

*Draft writeup for Phase 3d. Target: dev.to, HN Show, Vercel AI SDK Discord. ~1500 words.*

---

## The headline

A JavaScript function call. One query. 10,000 MS-MARCO passages. Exact cosine-similarity top-10.

**0.06 ms.** Recall: 1.0.

That's a Node.js native addon calling a cached GPU matmul written in Mojo, running on an H100. 19,000 queries per second from a single process, p99 = 0.09 ms, no Python runtime, no hosted vector DB, no network hop. The corpus lives on the GPU; each query ships ~1.5 KB up, gets a `[1, N]` score matrix computed in registers, and the top-k reduction fuses into the same kernel so the D2H transfer is 40 bytes instead of 40 KB.

This post is about how that number got honest, what it doesn't mean, and the ~80 lines of code you'd actually run.

---

## The honest table

Same shape — `[1, 384] × [384, 10000]`, real MS-MARCO passages embedded with `Xenova/all-MiniLM-L6-v2`, queried against the actual MS-MARCO query set. k=10, recall measured against exact ground truth:

**H100 80 GB HBM3:**

| Baseline | Latency | Recall@10 | |
| --- | ---: | ---: | --- |
| HNSW `ef=100` | 0.20 ms | 1.00 | |
| HNSW `ef=500` | 0.68 ms | 1.00 | |
| HNSW `ef=2000` | 1.79 ms | 1.00 | |
| ORT CPU | 0.13 ms | 1.00 | onnxruntime-node |
| **GPU `searchHandle`** | **0.06 ms** | **1.00** | Mojo + `linalg.matmul` + fused top-k |

**M4 Metal (Apple Silicon):**

| Baseline | Latency | Recall@10 | |
| --- | ---: | ---: | --- |
| HNSW `ef=100` | 0.26 ms | 1.00 | HNSW wins single-query on M4 |
| ORT CPU | 1.32 ms | 1.00 | |
| **GPU `searchHandle`** | **3.59 ms** | **1.00** | |

Two tables because one headline hides a tradeoff. On H100 the GPU path dominates. On M4 the Metal dispatch tax eats the win — HNSW `ef=100` is 13× faster for single-query. That flips back at batch-64 on M4 (GPU is 1.3–11× faster than HNSW because it amortizes the fixed kernel cost). So the real story is: **the GPU path wins where the GPU is fast** — H100 across the board, M4 for batched workloads.

And one more row worth naming: at batch-64 on H100, GPU exact is **9–23× faster** than HNSW at ef=100/500/2000. At batch-256 × 100K corpus (synthetic random vectors, where HNSW recall craters to 0.07), GPU exact is **113× faster than HNSW ef=2000**. The ANN tradeoff evaporates when you have a GPU and a batch.

---

## Why this was hard to get honest

The first version of this bench used random unit vectors. That made HNSW look terrible: recall@10 at ef=100 was 0.10, ef=2000 was 0.65. Graph-based ANN is pathological on uniform random vectors — concentration of measure in high dimensions makes "top-10 nearest" a near-tie among thousands of candidates, and HNSW's proximity graph can't discriminate. Shipping those numbers as the comparison would have been technically correct and morally dishonest. So the first step in writing this post was building a real-embedding fixture:

1. Fetch 10k MS-MARCO passages + 200 queries from the HF datasets-server REST API (paginated, cached per-page so rate-limits don't wipe a partial fetch).
2. Embed them with `@xenova/transformers` running `Xenova/all-MiniLM-L6-v2` on CPU — d=384, mean-pooled, L2-normalized.
3. Write two `.bin` files with a tiny header (magic + dtype + d + n + float32 data).

With real embeddings, HNSW recall@10 is 1.00 at ef=100. That's the correct baseline to compare against, and it's the baseline in the tables above. No cherry-picking.

Everything in this post is reproducible with:

```bash
node scripts/build-msmarco-fixture.js              # ~10 min, one-time
bash scripts/runpod-bench-3d.sh                    # H100 on RunPod, ~20 min, ~$1
# or locally:
node matmul/matmul_rag.js --fixture=msmarco-10k
```

Raw captures are in the repo at [`docs/bench-rag-3d-h100-msmarco.txt`](bench-rag-3d-h100-msmarco.txt) and [`docs/bench-rag-3d-msmarco-m4.txt`](bench-rag-3d-msmarco-m4.txt).

---

## The 80-line demo

The interesting code is in [`examples/rag-demo/search.js`](../examples/rag-demo/search.js). Here's the whole public surface:

```javascript
const cached = require('./matmul/build/matmul_cached.node');

class GpuIndex {
  constructor({ docs, embeddings, dim }) {
    this.docs = docs;
    this.dim = dim;
    this.n = docs.length;
    // Transpose [N, dim] → [dim, N] so query @ corpus gives [1, N] scores.
    const corpusT = new Float32Array(dim * this.n);
    for (let i = 0; i < this.n; i++) {
      for (let j = 0; j < dim; j++) corpusT[j * this.n + i] = embeddings[i * dim + j];
    }
    this.hCorpus = cached.loadMatrixGpu(corpusT, dim, this.n);
  }

  search(queryEmbedding, k) {
    const hQuery = cached.loadMatrixGpu(queryEmbedding, 1, this.dim);
    const idx = new Uint32Array(k);
    const scores = new Float32Array(k);
    cached.searchHandle(hQuery, this.hCorpus, idx, scores);
    cached.releaseMatrixGpu(hQuery);
    return Array.from(idx, (i, r) => ({ doc: this.docs[i], score: scores[r], index: i }));
  }

  close() { cached.releaseMatrixGpu(this.hCorpus); }
}
```

That's the whole API. `loadMatrixGpu` is a one-time H2D copy; the returned handle is a persistent GPU buffer you reuse across queries. `searchHandle` runs the matmul + top-k on the GPU and writes the top-k indices and scores back to JS-owned `Uint32Array` / `Float32Array` — no per-query allocation on the hot path, no JSON boundary, no IPC.

What's happening on the GPU is boring in a good way. It's `linalg.matmul` — Mojo's stdlib matmul, which lowers to tensor-core MMA instructions on H100. The top-k reduction is a straight scan over the `[1, N]` score matrix in registers; for k=10 and N=10k that's a single threadblock of work after the matmul finishes. No custom kernel, no hand-tuned shared-memory dance. The speedup over a custom tall-skinny kernel we tried earlier was... negative; `linalg.matmul` is already fast enough that the optimization surface is elsewhere. There's a [commit note](https://github.com/codetalcott/mojo-addon-examples/commit/54d54bc) on that.

---

## What this is not

- **It's not a vector DB.** No metadata filters, no persistence, no sharding, no multi-tenancy, no durability. It's a matmul and a top-k. If you need anything a real vector DB does, use one.
- **It's not faster than HNSW on CPU.** On M4, HNSW `ef=100` beats the GPU path by 13× for single-query on this corpus size. If you don't have a GPU, `hnswlib-node` is the right answer.
- **It doesn't beat ORT CPU for batched matmul on Apple Silicon.** `onnxruntime-node`'s CPU MatMul at batch-64 × 10k on M4 runs at 268 GFLOP/s — about 2× the GPU cached path. Apple's CPU SIMD is extraordinary; Metal's dispatch overhead is real.
- **There's a p99 wart.** At small-shape bursts on H100 (`[1, 768] × 100k`, not the fixture shape), p50 = 0.17 ms but p99 = 77 ms — a 443× spike every ~100 calls. p95 is clean. Looks like a CUDA stream or driver hiccup that doesn't reproduce at larger shapes (p99/p50 = 1.0× at 1M corpus). If you deploy this in a latency-sensitive path, measure your own p99 and plan around it. For offline/batch workloads it's irrelevant.
- **It's not a product (yet).** This is a working prototype in a research repo. No npm install, no docs beyond the README, no SLAs. If it's useful, let me know what form you'd want it in.

---

## Who this is for

If you're building a RAG app in Node.js, have a corpus that fits on one GPU (call it < 10M vectors × 768 dim = ~30 GB), and a workload that can tolerate owning an H100 (or sharing one), the exact-retrieval path is now on the table. No Python bridge, no vector DB licence, no cross-region network hop. The corpus lives in HBM3, the query is a ~1.5 KB H2D, and the answer is back in 60 µs with perfect recall.

The "fits on one GPU" line is the constraint that matters. At `[1, 768] × 1M` the H100 bench shows sub-2 ms single-query at recall=1.0 — that's the end of the paved road for now. Past that you need to shard, and this repo doesn't.

If you're building in Python, none of this applies; you already have FAISS, which does all of this and more. The point here is that **the same capability is now reachable from Node**, with ~300 lines of Mojo + napi glue, built from a public stdlib matmul and a public N-API binding pattern.

---

## Try it in 20 minutes for ~$1

```bash
# On a fresh RunPod H100 pod (H100 PCIe 80GB, ~$2/hr):
git clone https://github.com/codetalcott/mojo-addon-examples
cd mojo-addon-examples
FIXTURE=1 bash scripts/runpod-bench-3d.sh
cat ~/bench-rag-3d.txt
```

That script installs pixi + Node, builds the cached-matmul addon, runs the regression, builds the MS-MARCO fixture, and runs the whole bench suite (synthetic + real embeddings, single-query + batch-64 + batch-256, 100k + 1M corpus). Terminate the pod when it's done. Under a dollar.

Code: [github.com/codetalcott/mojo-addon-examples](https://github.com/codetalcott/mojo-addon-examples).

Feedback wanted. If the right next move is a standalone `@org/node-rag` package, that's a conversation I'd like to have.
