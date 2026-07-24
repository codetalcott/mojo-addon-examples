# @qkstat/retrieve

GPU-accelerated exact-retrieval RAG primitives for Node.js, built in Mojo.

Cached GPU matmul + fused host-side top-k — the building blocks for semantic search against a GPU-resident corpus. No approximate-nearest-neighbor tradeoff: recall is 1.0 by construction.

## Status

**Experimental (v0.1.0-pre).** API is stabilizing. Distribution gated on Modular license review for bundled MAX runtime redistribution — source build works today on any host with Mojo + pixi installed. See the [parent repo README](../../README.md) for the full story.

## Install

```bash
npm install @qkstat/retrieve
```

Platform prebuilts (not yet published to npm):

- `@qkstat/retrieve-darwin-arm64` — Apple Silicon (Metal)
- `@qkstat/retrieve-linux-x64` — Linux x86_64 (NVIDIA, any compute capability ≥ 8.0 via driver PTX JIT)

Until prebuilts ship, build from source from the monorepo:

```bash
git clone https://github.com/codetalcott/mojo-addon-examples
cd mojo-addon-examples
npm install                                        # pulls napi-mojo framework
pixi run bash packages/retrieve/build.sh                # builds packages/retrieve/build/retrieve.node
```

## Use

```js
const { GpuIndex } = require('@qkstat/retrieve');

// embeddings is a Float32Array of length docs.length * dim, row-major.
// L2-normalize each row for cosine similarity.
const index = new GpuIndex({ docs, embeddings, dim: 384 });

const query = /* Float32Array of length dim */;
const top10 = index.search(query, 10);
// [{ doc, score, index }, ...] sorted descending by score

// When done:
index.close();
```

Or use the raw primitives directly:

```js
const { loadMatrixGpu, matmulHandle, searchHandle, releaseMatrixGpu } = require('@qkstat/retrieve');

const hCorpus = loadMatrixGpu(corpusColMajor, dim, N);  // one-time H2D
const hQuery = loadMatrixGpu(queryRowMajor, B, dim);
const idx = new Uint32Array(B * k);
const scores = new Float32Array(B * k);
searchHandle(hQuery, hCorpus, idx, scores);             // fused matmul + top-k
releaseMatrixGpu(hQuery);
// ... loop, reusing hCorpus ...
releaseMatrixGpu(hCorpus);
```

`searchHandle` is the fast path: it fuses the matmul and the per-row top-k so only `[B, k]` indices + scores are shipped back to the host, not the full `[B, N]` score matrix.

## Benchmarks

At `[1, 384] × [384, 10k]` against real MS-MARCO embeddings (MiniLM-L6-v2):

| Hardware | Latency | Recall@10 |
| --- | ---: | ---: |
| H100 80GB HBM3 | **0.06 ms** | **1.00** |
| M4 Metal | 3.59 ms | 1.00 |

M4 Metal has no FP32 tensor cores; HNSW `ef=100` beats the GPU path there by 13×. On H100 the order flips — GPU exact wins 3.3–29× over HNSW at every recall level. The reproducer lives at [`examples/matmul/matmul_rag.js`](../../examples/matmul/matmul_rag.js) in the parent repo; full M4 + H100 captures under [`docs/`](../../docs/).

## Requirements

- **Node.js ≥ 22.12** (N-API v10)
- **NVIDIA driver ≥ 580** (for the bundled PTX driver wrapper) OR macOS 14+ with Apple Silicon
- No CUDA toolkit, no Python, no Mojo/MAX runtime required on the end user's box once prebuilts ship

## License

MIT. See [LICENSE](../../LICENSE) in the parent repo.
