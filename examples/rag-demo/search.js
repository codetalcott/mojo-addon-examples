// examples/rag-demo/search.js — local RAG search in ~80 lines of Node.
//
// Demonstrates the full retrieval loop on cached GPU matmul + fused top-k:
//   1. Load precomputed embeddings for a corpus of documents (one-time H2D).
//   2. For each query embedding, searchHandle runs cosine-similarity scoring
//      against the whole corpus on the GPU and returns the top-k indices
//      + scores in one call.
//
// The demo generates a synthetic corpus (10k random documents with random
// 768-dim "embeddings") so it runs standalone. To use real embeddings, swap
// `makeSyntheticCorpus` for your own loader: any `{docs: string[], embeddings:
// Float32Array}` pair works, embeddings should be L2-normalized if you want
// the raw matmul output to equal cosine similarity.
//
// Run: node examples/rag-demo/search.js
// Flags: --corpus=N (default 10000), --dim=D (default 768), --k=K (default 10)

const path = require('path');

const CACHED_PATH = path.resolve(__dirname, '../../matmul/build/matmul_cached.node');
let cached;
try {
  cached = require(CACHED_PATH);
} catch (e) {
  console.error('matmul_cached.node not found — run: pixi run bash matmul/build_cached.sh');
  process.exit(1);
}

function parseFlag(name, fallback) {
  const a = process.argv.find((s) => s.startsWith(`--${name}=`));
  return a ? parseInt(a.split('=')[1], 10) : fallback;
}

const N = parseFlag('corpus', 10_000);
const DIM = parseFlag('dim', 768);
const K = parseFlag('k', 10);

// --- Synthetic corpus --------------------------------------------------------
// Real use case: replace this with a loader for your precomputed embeddings.

function makeSyntheticCorpus(n, dim) {
  const docs = new Array(n);
  const embeddings = new Float32Array(n * dim);
  for (let i = 0; i < n; i++) {
    docs[i] = `Document #${i} — synthetic fixture`;
    // Random unit vector (approximate — not true L2 normalization).
    let norm = 0;
    for (let j = 0; j < dim; j++) {
      const v = Math.random() * 2 - 1;
      embeddings[i * dim + j] = v;
      norm += v * v;
    }
    norm = Math.sqrt(norm);
    for (let j = 0; j < dim; j++) embeddings[i * dim + j] /= norm;
  }
  return { docs, embeddings };
}

// --- Index ------------------------------------------------------------------

class GpuIndex {
  constructor({ docs, embeddings, dim }) {
    this.docs = docs;
    this.dim = dim;
    this.n = docs.length;

    // Corpus is stored as [dim, N] so `query @ corpus` gives [1, N] scores.
    // The caller passed [N, dim] row-major — transpose on load so searchHandle
    // sees the corpus side as B.cols = N.
    const corpusT = new Float32Array(dim * this.n);
    for (let i = 0; i < this.n; i++) {
      for (let j = 0; j < dim; j++) {
        corpusT[j * this.n + i] = embeddings[i * dim + j];
      }
    }
    this.hCorpus = cached.loadMatrixGpu(corpusT, dim, this.n);
  }

  search(queryEmbedding, k) {
    // queryEmbedding shape is [dim]; treat as [1, dim].
    const hQuery = cached.loadMatrixGpu(queryEmbedding, 1, this.dim);
    const idx = new Uint32Array(k);
    const scores = new Float32Array(k);
    cached.searchHandle(hQuery, this.hCorpus, idx, scores);
    cached.releaseMatrixGpu(hQuery);

    const results = new Array(k);
    for (let i = 0; i < k; i++) {
      results[i] = { doc: this.docs[idx[i]], score: scores[i], index: idx[i] };
    }
    return results;
  }

  close() {
    cached.releaseMatrixGpu(this.hCorpus);
  }
}

// --- Demo run ---------------------------------------------------------------

console.log(`Building synthetic corpus: ${N} docs × ${DIM} dim...`);
const tBuild = performance.now();
const { docs, embeddings } = makeSyntheticCorpus(N, DIM);
console.log(`  (generation: ${(performance.now() - tBuild).toFixed(0)}ms)`);

const tLoad = performance.now();
const index = new GpuIndex({ docs, embeddings, dim: DIM });
console.log(`Corpus uploaded to GPU: ${(performance.now() - tLoad).toFixed(0)}ms`);

// Use the 42nd doc's embedding as a query — its own entry should top the
// ranking for a sanity check on the search path.
const queryDoc = 42;
const query = embeddings.subarray(queryDoc * DIM, (queryDoc + 1) * DIM);

// Warm + time.
for (let i = 0; i < 3; i++) index.search(query, K);
const iters = 50;
const tSearch = performance.now();
let results;
for (let i = 0; i < iters; i++) results = index.search(query, K);
const perCall = (performance.now() - tSearch) / iters;

console.log(`\nQuery: embedding of doc #${queryDoc}`);
console.log(`Top ${K} results (avg ${perCall.toFixed(2)}ms/query over ${iters} iters):`);
for (let i = 0; i < K; i++) {
  const r = results[i];
  console.log(`  ${String(i + 1).padStart(2)}. score=${r.score.toFixed(4)} idx=${r.index} :: ${r.doc}`);
}

index.close();
