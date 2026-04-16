// spike/demo-reference.js — pure CPU reference for Day 4 demo comparison.
//
// Same workload as spike/demo.js:
//   - 1000-doc programmatic clustered corpus (shared via corpus.js)
//   - embed + index + 10 queries
//
// Implementation:
//   - @huggingface/transformers (ONNX Runtime CPU) for embedding
//   - hnswlib-node (CPU) for approximate search, swept over ef ∈ {100, 500, 2000}
//
// No MAX, no Mojo, no GPU. Used for the head-to-head comparison numbers
// in the Day 10 writeup.

const { buildCorpus, buildQueries } = require('./corpus');

const EMBED_DIM = 384;
const K = 10;
const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const EF_VALUES = [100, 500, 2000];

async function main() {
  console.log('loading @huggingface/transformers feature-extraction pipeline (CPU, unquantized)...');
  const { pipeline } = await import('@huggingface/transformers');
  const tExtStart = performance.now();
  const extractor = await pipeline('feature-extraction', MODEL_ID, { quantized: false });
  console.log(`pipeline ready: ${(performance.now() - tExtStart).toFixed(0)}ms`);

  console.log('\nrequiring hnswlib-node...');
  let hnswlib;
  try {
    hnswlib = require('hnswlib-node');
  } catch (e) {
    console.error('hnswlib-node not installed. Run: cd spike && npm install hnswlib-node');
    throw e;
  }

  console.log('\nbuilding corpus...');
  const { docs, clusters, clusterNames } = buildCorpus(200);
  console.log(`corpus: ${docs.length} docs`);

  // --- Embed corpus ------------------------------------------------------
  console.log('\nembedding corpus on CPU (this is the slow part)...');
  const tEmbStart = performance.now();
  const BATCH = 32;
  const embeddings = new Float32Array(docs.length * EMBED_DIM);
  for (let off = 0; off < docs.length; off += BATCH) {
    const batchDocs = docs.slice(off, off + BATCH);
    const out = await extractor(batchDocs, { pooling: 'mean', normalize: true });
    embeddings.set(out.data, off * EMBED_DIM);
  }
  const embMs = performance.now() - tEmbStart;
  console.log(`corpus embed time: ${embMs.toFixed(0)}ms (${(docs.length * 1000 / embMs).toFixed(0)} docs/sec)`);

  // --- Build HNSW index --------------------------------------------------
  console.log('\nbuilding HNSW index (M=16, efConstruction=100, ip space)...');
  const tIdxStart = performance.now();
  const index = new hnswlib.HierarchicalNSW('ip', EMBED_DIM);
  index.initIndex(docs.length, 16, 100, 100);
  for (let i = 0; i < docs.length; i++) {
    index.addPoint(Array.from(embeddings.subarray(i * EMBED_DIM, (i + 1) * EMBED_DIM)), i);
  }
  console.log(`HNSW build: ${(performance.now() - tIdxStart).toFixed(0)}ms`);

  // --- Queries -----------------------------------------------------------
  const { queries, queryClusterTruth } = buildQueries(2);
  console.log(`\nrunning ${queries.length} queries (k=${K})...`);

  // Warmup — embed + search to prime JIT and caches
  {
    const w = await extractor([queries[0]], { pooling: 'mean', normalize: true });
    index.setEf(100);
    index.searchKnn(Array.from(w.data), K);
  }

  const results = {};
  for (const ef of EF_VALUES) {
    index.setEf(ef);
    const embedLatencies = [];
    const searchLatencies = [];
    const queryLatencies = [];
    let totalHits = 0;
    for (let qi = 0; qi < queries.length; qi++) {
      const t0 = performance.now();
      const qEmb = await extractor([queries[qi]], { pooling: 'mean', normalize: true });
      const tEmb = performance.now();
      const res = index.searchKnn(Array.from(qEmb.data), K);
      const tSearch = performance.now();

      embedLatencies.push(tEmb - t0);
      searchLatencies.push(tSearch - tEmb);
      queryLatencies.push(tSearch - t0);

      const inCluster = res.neighbors.filter((ni) => clusters[ni] === queryClusterTruth[qi]).length;
      totalHits += inCluster;
    }

    const avgL = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const p50 = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length * 0.5)];
    results[ef] = {
      hits: totalHits,
      hitPct: 100 * totalHits / (queries.length * K),
      embedP50: p50(embedLatencies),
      searchP50: p50(searchLatencies),
      totalP50: p50(queryLatencies),
      totalAvg: avgL(queryLatencies),
    };
  }

  console.log('\n==========================');
  console.log('SUMMARY — reference (CPU ONNX + hnswlib-node)');
  console.log('==========================');
  console.log(`corpus: ${docs.length} docs, ${clusterNames.length} clusters`);
  console.log(`corpus embed throughput: ${(docs.length * 1000 / embMs).toFixed(0)} docs/sec  (${embMs.toFixed(0)}ms total)`);
  console.log('');
  console.log('ef    in-cluster%    embed p50   search p50   total p50    total avg');
  for (const ef of EF_VALUES) {
    const r = results[ef];
    console.log(
      `${String(ef).padStart(4)}  ${r.hitPct.toFixed(1).padStart(6)}%       `
      + `${r.embedP50.toFixed(2).padStart(7)}ms    `
      + `${r.searchP50.toFixed(3).padStart(8)}ms   `
      + `${r.totalP50.toFixed(2).padStart(7)}ms    `
      + `${r.totalAvg.toFixed(2).padStart(7)}ms`);
  }

  console.log('\nDONE');
}

main().catch((e) => { console.error('FAIL:', e); process.exit(1); });
