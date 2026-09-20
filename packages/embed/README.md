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

- `@qkstat/embed-darwin-arm64` — Apple Silicon (**no useful GPU path yet** — see [Apple Silicon](#apple-silicon))
- `@qkstat/embed-linux-x64` — Linux x86_64 (NVIDIA sm_80+; single binary covers H100/H200 via driver PTX JIT)

Until prebuilts ship, build from source from the monorepo:

```bash
git clone https://github.com/codetalcott/mojo-addon-examples
cd mojo-addon-examples
npm install                                        # pulls napi-mojo framework
pixi run bash packages/embed/build.sh              # builds packages/embed/build/embed.node
```

## Use

End-to-end RAG in three lines via the bundled `RagPipeline` (composes embed + [`@qkstat/retrieve`](../retrieve) for you):

```js
const { RagPipeline } = require('@qkstat/embed');

const pipe = new RagPipeline();
await pipe.warmup();                          // pays MAX cold-start; idempotent
await pipe.addTexts(docs);                    // embed corpus + build GpuIndex
const hits = await pipe.search('how does auth work', 10);
// → [{ doc, score, index }, ...] sorted desc

pipe.close();
```

`RagPipeline` loads both `embed.node` and `retrieve.node` into the same Node process on separate CUDA contexts. `warmup()` is the right place to absorb the ~30 s MAX graph compile in a long-lived service (MCP daemon, watch-mode reindexer); after that, queries are warm-path.

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
const { GpuIndex } = require('@qkstat/retrieve');

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
- **NVIDIA driver ≥ 580** for the GPU path. Apple Silicon runs, but at CPU-class speed — see [Apple Silicon](#apple-silicon)
- Model weights auto-downloaded via HuggingFace on first run (cached to `HF_HOME`)

## Apple Silicon

The package runs on an M-series Mac and is numerically correct there (cosine 1.000000 vs the CPU reference). As of MAX 26.6.0 `driver.Accelerator()` succeeds on an M4 and reports `Device(type=gpu,id=0)`; earlier releases raised "Not implemented for device: Apple M4" and fell back to CPU.

**The `gpu` device is the right default on Apple Silicon, but the margin is small and depends on sequence length.** Measured on an M4, `execute` + explicit `synchronize()`, n=8 after warmup:

| shape (batch × seq) | tokens | `gpu` | `cpu` | winner |
| --- | --- | --- | --- | --- |
| 1×14 | 14 | 5.02 ms | 3.27 ms | CPU 1.54× |
| 8×14 | 112 | 5.37 ms | 5.61 ms | GPU 1.04× |
| **100×14** | **1 400** | **25.75 ms** | **30.74 ms** | **GPU 1.19×** |
| 32×32 | 1 024 | 19.57 ms | 25.54 ms | GPU 1.30× |
| 100×32 | 3 200 | 56.69 ms | 61.85 ms | GPU 1.09× |
| 64×64 | 4 096 | 83.05 ms | 91.61 ms | GPU 1.10× |
| 128×64 | 8 192 | 163.05 ms | 170.36 ms | GPU 1.04× |
| 256×128 | 32 768 | 836.15 ms | 722.02 ms | CPU 1.16× |

The GPU leads across the usual range and loses at the two extremes: single short inputs, where dispatch overhead dominates, and long sequences, where O(S²) attention scales worse on this backend. The crossover tracks **sequence length, not batch size** — chunking a 256×128 batch into smaller batches at the same seq_len does not recover the win. MiniLM-L6 is a short-sequence model, so normal use sits in the GPU-favouring region.

Treat this as parity-class either way. It is not H100 acceleration — see [Benchmarks](#benchmarks) — so benchmark and deploy on NVIDIA; Apple Silicon is for correctness work and local iteration.

Things that do *not* help on M4, all measured rather than assumed: **fp16** (961.7 ms vs 912.7 ms at 256×128 — 0.95×, no gain, so arithmetic throughput is not the constraint), **batch chunking** (above), and **zero-copy transfer** — the D2H is already 0.4–0.8 ms via DLPack against ~900 ms of compute, so the "Phase 2 zero-copy" note in `embed.py` has nothing to win here. If that optimization is worth doing, the case has to come from H100 and real PCIe.

### `EMBED_REQUIRE_GPU`

The library falls back to `driver.CPU()` when `Accelerator()` fails, which keeps it usable on hosts with no accelerator but means a GPU regression can pass a test suite unnoticed. Set `EMBED_REQUIRE_GPU=1` to make that fallback a hard error instead:

```bash
EMBED_REQUIRE_GPU=1 node your-script.js
```

The repo's `scripts/verify-all.sh` defaults it to `1` for its own runs on every platform; `EMBED_REQUIRE_GPU=0` opts back out. On Apple Silicon this proves the device initialized and, per the numbers above, that you are on the faster of the two paths at typical shapes — but the margin is small, so do not read it as evidence of GPU-class speedup.

## License

MIT. See [LICENSE](../../LICENSE) in the parent repo. Vendored BERT pipeline code (`bert_graph.py`, `bert_weight_adapter.py`) is Apache-2.0 from Modular's `max.pipelines.architectures.bert`.
