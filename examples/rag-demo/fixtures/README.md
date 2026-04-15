# MS-MARCO real-embedding fixtures

This directory holds pre-computed sentence embeddings used by the matmul RAG
benchmark when invoked with `--fixture=msmarco-10k`. Files are not committed
to git — generate them locally with the builder script.

## Build

```bash
node scripts/build-msmarco-fixture.js
# produces:
#   examples/rag-demo/fixtures/msmarco-10k-corpus.bin   (~15 MB)
#   examples/rag-demo/fixtures/msmarco-10k-queries.bin  (~300 KB)
```

The first run pulls the first N passages and Q queries from BeIR's
MS-MARCO via the HuggingFace datasets-server REST API (paginated, ~100
small requests for N=10000) and caches the raw text under `.cache/`. Plus
~25 MB of MiniLM-L6-v2 model weights to `~/.cache/huggingface/`. Both are
reused on subsequent runs.

Flags: `--n=10000` (corpus size), `--queries=200` (query count),
`--out=examples/rag-demo/fixtures` (output dir).

Embedding model is `Xenova/all-MiniLM-L6-v2` (d=384), mean-pooled and
L2-normalized — drop-in compatible with the `b = makeMatrix(d, N)` corpus
the synthetic bench uses.

## Format

```text
offset  bytes  field
0       4      magic   "MMR1"
4       4      dtype   uint32 LE (0 = float32)
8       4      d       uint32 LE
12      4      n       uint32 LE
16      n*d*4  data    float32 LE, row-major (one row per vector)
```
