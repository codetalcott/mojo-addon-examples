// spike/bench.js — Day 5 warm-path benchmarks.
//
// Measures the spike's embed + search pipeline across production-ish shapes
// and reports honest p50/p95/p99 per stage (tokenize / embed / search /
// total). Cold CUDA JIT is explicitly excluded via a warmup phase.
//
// Shapes:
//   - query batch:    1, 8, 64
//   - query seq_len:  32, 128 (tokenizer-cap 128)
//   - corpus size:    1000, 10000
//
// Compare against spike/demo-reference.js (CPU ONNX + hnswlib) for the
// head-to-head comparison that shows up in the Day 10 writeup.

const path = require('path');
const { tokenize } = require('./tokenize');
const { buildCorpus, buildQueries } = require('./corpus');
const { GpuIndex } = require('../packages/rag/lib/GpuIndex');

const EMBED_DIM = 384;
const K = 10;
const WARMUP = 10;
const ITERS = 50;

function pct(arr, p) {
  const s = [...arr].sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.floor(s.length * p));
  return s[i];
}
const p50 = (a) => pct(a, 0.50);
const p95 = (a) => pct(a, 0.95);
const p99 = (a) => pct(a, 0.99);
const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;

// Generate N corpus docs with controlled rough seq_len via doc length.
function genCorpus(n) {
  const { docs, clusterNames } = buildCorpus(Math.ceil(n / 5));
  return { docs: docs.slice(0, n), clusterNames };
}

// Generate B queries of specific target seq_len via string padding/slicing.
// We want tokenizer to produce roughly the target seq_len after WordPiece.
function genQueries(n, targetLen) {
  // Rough heuristic: ~1.2 tokens per word for English.
  const targetWords = Math.max(1, Math.floor(targetLen / 1.2) - 2); // -2 for [CLS]+[SEP]
  const { queries } = buildQueries(Math.ceil(n / 5));
  return queries.slice(0, n).map((q) => {
    // Expand short queries to target length by appending generic filler.
    const words = q.split(/\s+/);
    while (words.length < targetWords) words.push(...q.split(/\s+/));
    return words.slice(0, targetWords).join(' ');
  });
}

async function embedBatch(addon, docs) {
  const t0 = performance.now();
  const { ids, mask, batch, seqLen } = await tokenize(docs);
  const t1 = performance.now();
  const dst = new Float32Array(batch * EMBED_DIM);
  addon.embedTokens(ids, mask, batch, seqLen, dst);
  const t2 = performance.now();
  return { embeddings: dst, batch, seqLen, tokMs: t1 - t0, embMs: t2 - t1 };
}

async function benchQueryShape(addon, index, queries, label) {
  // Warmup
  for (let i = 0; i < WARMUP; i++) {
    const { embeddings } = await embedBatch(addon, queries);
    // For batch>1, search each query separately to match real serving.
    for (let j = 0; j < embeddings.length / EMBED_DIM; j++) {
      const q = embeddings.subarray(j * EMBED_DIM, (j + 1) * EMBED_DIM);
      index.search(q, K);
    }
  }

  const tokMs = [], embMs = [], searchMs = [], totalMs = [];
  for (let i = 0; i < ITERS; i++) {
    const t0 = performance.now();
    const { embeddings, tokMs: tM, embMs: eM } = await embedBatch(addon, queries);
    const t1 = performance.now();
    const numQ = embeddings.length / EMBED_DIM;
    for (let j = 0; j < numQ; j++) {
      const q = embeddings.subarray(j * EMBED_DIM, (j + 1) * EMBED_DIM);
      index.search(q, K);
    }
    const t2 = performance.now();
    tokMs.push(tM);
    embMs.push(eM);
    searchMs.push(t2 - t1);
    totalMs.push(t2 - t0);
  }

  console.log(
    `  ${label.padEnd(22)} ` +
    `tok=${p50(tokMs).toFixed(2).padStart(5)}ms ` +
    `emb=${p50(embMs).toFixed(2).padStart(5)}ms ` +
    `search=${p50(searchMs).toFixed(3).padStart(6)}ms ` +
    `total=${p50(totalMs).toFixed(2).padStart(5)}ms ` +
    `(p95=${p95(totalMs).toFixed(2)}ms p99=${p99(totalMs).toFixed(2)}ms)`);
  return { tokMs, embMs, searchMs, totalMs };
}

async function benchCorpusEmbed(addon, docs, label) {
  // Warmup: first batch pays CUDA kernel JIT; we want to report post-JIT throughput.
  const BATCH = 64;
  console.log(`  building ${docs.length}-doc corpus embeddings (warming up JIT)...`);
  const all = new Float32Array(docs.length * EMBED_DIM);
  const batchTimes = [];

  for (let off = 0; off < docs.length; off += BATCH) {
    const batchDocs = docs.slice(off, off + BATCH);
    const t0 = performance.now();
    const { embeddings } = await embedBatch(addon, batchDocs);
    const t1 = performance.now();
    all.set(embeddings, off * EMBED_DIM);
    batchTimes.push(t1 - t0);
  }

  // Drop the first batch (cold JIT).
  const warmTimes = batchTimes.slice(1);
  const warmTotal = warmTimes.reduce((a, b) => a + b, 0);
  const warmDocs = (batchTimes.length - 1) * BATCH;
  const warmQPS = warmDocs / (warmTotal / 1000);
  console.log(
    `  ${label.padEnd(22)} ` +
    `cold_first_batch=${batchTimes[0].toFixed(0)}ms  ` +
    `warm_per_batch_p50=${p50(warmTimes).toFixed(2)}ms  ` +
    `warm_throughput=${warmQPS.toFixed(0)} docs/sec`);
  return all;
}

async function main() {
  const addon = require(path.join(__dirname, 'build', 'embed.node'));
  if (typeof addon.embedTokens !== 'function') {
    throw new Error('embed.node missing — run spike/build.sh');
  }

  // Corpus sizes to test
  const CORPUS_SIZES = [1000, 10000];
  // Query shapes to test (batch × target_seq_len)
  const QUERY_SHAPES = [
    { B: 1, S: 32,  label: 'batch-1  seq-32' },
    { B: 1, S: 128, label: 'batch-1  seq-128' },
    { B: 8, S: 32,  label: 'batch-8  seq-32' },
    { B: 64, S: 32, label: 'batch-64 seq-32' },
    { B: 64, S: 128, label: 'batch-64 seq-128' },
  ];

  console.log(`spike bench  —  warmup=${WARMUP} iters, measure=${ITERS} iters\n`);

  for (const N of CORPUS_SIZES) {
    console.log(`\n==================================================`);
    console.log(`CORPUS ${N} docs`);
    console.log(`==================================================`);

    const { docs } = genCorpus(N);

    // Build + measure corpus-embed throughput
    console.log('\n[1] Corpus embedding (batch-64):');
    const embeddings = await benchCorpusEmbed(addon, docs, `${N} docs`);

    // Build index
    console.log('\n[2] Building GpuIndex...');
    const tIdx = performance.now();
    const index = new GpuIndex({ docs, embeddings, dim: EMBED_DIM });
    console.log(`  GpuIndex build: ${(performance.now() - tIdx).toFixed(1)}ms (H2D of ${(N * EMBED_DIM * 4 / 1e6).toFixed(1)} MB)`);

    // Query latencies across shapes
    console.log('\n[3] Query latencies (WARM, median over 50 iters):');
    for (const { B, S, label } of QUERY_SHAPES) {
      const queries = genQueries(B, S);
      await benchQueryShape(addon, index, queries, label);
    }

    index.close();
  }

  console.log('\nDONE');
}

main().catch((e) => { console.error('FAIL:', e); process.exit(1); });
