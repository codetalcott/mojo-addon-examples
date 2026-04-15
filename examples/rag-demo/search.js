// examples/rag-demo/search.js — local RAG search in ~30 lines of Node.
//
// Uses @qkstat/rag's GpuIndex helper — a thin wrapper around the four GPU
// primitives (loadMatrixGpu / matmulHandle / searchHandle / releaseMatrixGpu)
// that handles the row-major→column-major transpose and returns result
// objects with doc, score, and index.
//
// Run: node examples/rag-demo/search.js
// Flags: --corpus=N (default 10000), --dim=D (default 768), --k=K (default 10)

const { GpuIndex } = require('@qkstat/rag');

function parseFlag(name, fallback) {
  const a = process.argv.find((s) => s.startsWith(`--${name}=`));
  return a ? parseInt(a.split('=')[1], 10) : fallback;
}

const N = parseFlag('corpus', 10_000);
const DIM = parseFlag('dim', 768);
const K = parseFlag('k', 10);

// --- Synthetic corpus -------------------------------------------------------
// Real use case: replace this with a loader for your precomputed embeddings.

function makeSyntheticCorpus(n, dim) {
  const docs = new Array(n);
  const embeddings = new Float32Array(n * dim);
  for (let i = 0; i < n; i++) {
    docs[i] = `Document #${i} — synthetic fixture`;
    // Random unit vector (approximate normalization).
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
