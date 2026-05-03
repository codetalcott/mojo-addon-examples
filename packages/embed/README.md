# @qkstat/embed

GPU-accelerated MiniLM-L6-v2 embeddings for Node.js — MAX on H100 via Mojo + N-API.

Mean-pooled, L2-normalized 384-dim embeddings that match `sentence-transformers/all-MiniLM-L6-v2` to six decimal places, callable from a Node.js process with no Python, no Docker, no model server.

## Status

**Experimental (v0.1.0-pre).** Productized from the [embedding-kernel spike](../../docs/embedding-kernel-spike-findings.md) (GO verdict: 1.44 ms p50 end-to-end on H100, 0.99999 cosine vs CPU reference). Distribution gated on Modular license review for bundled MAX runtime — source build works today on any host with Mojo + pixi installed.

## Install

```bash
npm install @qkstat/embed
```

Platform prebuilts (not yet published):

- `@qkstat/embed-darwin-arm64` — Apple Silicon (MAX currently falls back to CPU on M4; GPU paths ship when MAX adds Metal backend)
- `@qkstat/embed-linux-x64` — Linux x86_64 (NVIDIA sm_80+; single binary covers H100/H200 via driver PTX JIT)

Until prebuilts ship, build from source from the monorepo:

```bash
git clone https://github.com/codetalcott/mojo-addon-examples
cd mojo-addon-examples
npm install                                        # pulls napi-mojo framework
pixi run bash packages/embed/build.sh              # builds packages/embed/build/embed.node
```

## Use

End-to-end RAG in three lines via the bundled `RagPipeline` (composes embed + [`@qkstat/rag`](../rag) for you):

```js
const { RagPipeline } = require('@qkstat/embed');

const pipe = new RagPipeline();
await pipe.warmup();                          // pays MAX cold-start; idempotent
await pipe.addTexts(docs);                    // embed corpus + build GpuIndex
const hits = await pipe.search('how does auth work', 10);
// → [{ doc, score, index }, ...] sorted desc

pipe.close();
```

`RagPipeline` loads both `embed.node` and `rag.node` into the same Node process on separate CUDA contexts. `warmup()` is the right place to absorb the ~30 s MAX graph compile in a long-lived service (MCP daemon, watch-mode reindexer); after that, queries are warm-path.

Just embeddings, no index:

```js
const { EmbeddingEngine } = require('@qkstat/embed');

const engine = new EmbeddingEngine();
const embeddings = await engine.embed(['hello world', 'semantic search']);
// Float32Array of length 2 * 384, row-major, L2-normalized
```

Bring-your-own index (e.g. you already have a `GpuIndex` instance, or you want a different vector store):

```js
const { EmbeddingEngine } = require('@qkstat/embed');
const { GpuIndex } = require('@qkstat/rag');

const engine = new EmbeddingEngine();
const corpusEmb = await engine.embed(docs);
const index = new GpuIndex({ docs, embeddings: corpusEmb, dim: 384 });

const qEmb = await engine.embed([query]);
const top10 = index.search(qEmb, 10);
```

## Raw primitive

```js
const { embedTokens } = require('@qkstat/embed');

// ids, mask: Int32Array of shape [batch, seqLen]
// dst: Float32Array of shape [batch, 384] (pre-allocated, written in-place)
embedTokens(ids, mask, batch, seqLen, dst);
```

Tokenization is up to you — pair with `@huggingface/transformers` or bring your own WordPiece.

## Benchmarks

At H100 80GB HBM3, sentence-transformers/all-MiniLM-L6-v2:

| Shape | p50 | p95 |
| --- | ---: | ---: |
| batch-1, seq-32 | **1.44 ms** | 1.67 ms |
| batch-8, seq-32 | 1.87 ms | 2.15 ms |
| batch-64, seq-128 | 8.2 ms | 9.1 ms |

Numbers from the original spike on programmatically-clustered synthetic text. Real MS-MARCO numbers land in the next bench run — see [`docs/bench-embed-msmarco-*.txt`](../../docs/).

Cold start: ~29.6 s on first `embed()` call (MAX graph compile + CUDA JIT); subsequent calls are warm.

## Requirements

- **Node.js ≥ 22.12** (N-API v10)
- **NVIDIA driver ≥ 580** OR macOS 14+ with Apple Silicon (currently CPU fallback on M4)
- Model weights auto-downloaded via HuggingFace on first run (cached to `HF_HOME`)

## License

MIT. See [LICENSE](../../LICENSE) in the parent repo. Vendored BERT pipeline code (`bert_graph.py`, `bert_weight_adapter.py`) is Apache-2.0 from Modular's `max.pipelines.architectures.bert`.
