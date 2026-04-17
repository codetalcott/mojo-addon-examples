// packages/embed/bench.js — MS-MARCO warm-path benchmarks.
//
// Replaces the spike's synthetic clustered corpus with real MS-MARCO passages
// and queries (from examples/rag-demo/fixtures/msmarco-{N}k-{corpus,queries}.jsonl,
// produced by scripts/build-msmarco-fixture.js). Reports p50/p95/p99 per stage
// (tokenize / embed / search / total). Cold CUDA JIT is excluded via warmup.
//
// Shapes:
//   corpus size:    1000, 10000 (sampled from the 10k fixture)
//   query batch:    1, 8, 64
//   query seq_len:  natural MS-MARCO distribution, capped by tokenize.js
//                   MAX_SEQ_LEN (128)
//
// Run:  pixi run node packages/embed/bench.js
// Pre-req: fixtures built via
//   pixi run node scripts/build-msmarco-fixture.js --n=10000 --queries=200

const path = require('path');
const fs = require('fs');
const { tokenize } = require('./tokenize');
const { GpuIndex } = require('../rag/lib/GpuIndex');

const EMBED_DIM = 384;
const K = 10;
const WARMUP = 10;
const ITERS = 50;
const FIXTURE_DIR = path.resolve(__dirname, '..', '..', 'examples', 'rag-demo', 'fixtures');

function pct(arr, p) {
  const s = [...arr].sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.floor(s.length * p));
  return s[i];
}
const p50 = (a) => pct(a, 0.50);
const p95 = (a) => pct(a, 0.95);
const p99 = (a) => pct(a, 0.99);

function loadJsonl(p) {
  if (!fs.existsSync(p)) {
    throw new Error(
      `${p} missing. Build fixtures first:\n  pixi run node scripts/build-msmarco-fixture.js --n=10000 --queries=200`,
    );
  }
  const raw = fs.readFileSync(p, 'utf8');
  const texts = [];
  for (const line of raw.split('\n')) {
    if (!line) continue;
    texts.push(JSON.parse(line));
  }
  return texts;
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
  for (let i = 0; i < WARMUP; i++) {
    const { embeddings } = await embedBatch(addon, queries);
    for (let j = 0; j < embeddings.length / EMBED_DIM; j++) {
      const q = embeddings.subarray(j * EMBED_DIM, (j + 1) * EMBED_DIM);
      index.search(q, K);
    }
  }

  const tokMs = [], embMs = [], searchMs = [], totalMs = [];
  let seqLenObserved = 0;
  for (let i = 0; i < ITERS; i++) {
    const t0 = performance.now();
    const { embeddings, seqLen, tokMs: tM, embMs: eM } = await embedBatch(addon, queries);
    seqLenObserved = seqLen;
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
    `  ${label.padEnd(28)} seq=${String(seqLenObserved).padStart(3)} ` +
    `tok=${p50(tokMs).toFixed(2).padStart(5)}ms ` +
    `emb=${p50(embMs).toFixed(2).padStart(5)}ms ` +
    `search=${p50(searchMs).toFixed(3).padStart(6)}ms ` +
    `total=${p50(totalMs).toFixed(2).padStart(5)}ms ` +
    `(p95=${p95(totalMs).toFixed(2)}ms p99=${p99(totalMs).toFixed(2)}ms)`);
  return { tokMs, embMs, searchMs, totalMs, seqLen: seqLenObserved };
}

async function embedCorpus(addon, docs, label) {
  const BATCH = 64;
  console.log(`  embedding ${docs.length} passages in batch-${BATCH} (first batch = JIT)...`);
  const all = new Float32Array(docs.length * EMBED_DIM);
  const batchTimes = [];
  for (let off = 0; off < docs.length; off += BATCH) {
    const batchDocs = docs.slice(off, off + BATCH);
    const t0 = performance.now();
    const { embeddings } = await embedBatch(addon, batchDocs);
    batchTimes.push(performance.now() - t0);
    all.set(embeddings, off * EMBED_DIM);
  }
  const warmTimes = batchTimes.slice(1);
  const warmTotal = warmTimes.reduce((a, b) => a + b, 0);
  const warmDocs = (batchTimes.length - 1) * BATCH;
  const warmQPS = warmDocs / (warmTotal / 1000);
  console.log(
    `  ${label.padEnd(28)} ` +
    `cold_first_batch=${batchTimes[0].toFixed(0)}ms  ` +
    `warm_per_batch_p50=${p50(warmTimes).toFixed(2)}ms  ` +
    `warm_throughput=${warmQPS.toFixed(0)} docs/sec`);
  return all;
}

async function main() {
  const addon = require(path.join(__dirname, 'build', 'embed.node'));
  if (typeof addon.embedTokens !== 'function') {
    throw new Error('embed.node missing embedTokens — run packages/embed/build.sh');
  }

  const corpusTexts = loadJsonl(path.join(FIXTURE_DIR, 'msmarco-10k-corpus.jsonl'));
  const queryTexts = loadJsonl(path.join(FIXTURE_DIR, 'msmarco-10k-queries.jsonl'));
  console.log(`fixture: ${corpusTexts.length} passages, ${queryTexts.length} queries`);

  const CORPUS_SIZES = [1000, 10000];
  const QUERY_SHAPES = [
    { B: 1,  label: 'batch-1' },
    { B: 8,  label: 'batch-8' },
    { B: 64, label: 'batch-64' },
  ];

  console.log(`packages/embed bench (MS-MARCO) — warmup=${WARMUP} iters, measure=${ITERS} iters`);

  for (const N of CORPUS_SIZES) {
    console.log(`\n==================================================`);
    console.log(`CORPUS ${N} MS-MARCO passages`);
    console.log(`==================================================`);

    if (corpusTexts.length < N) {
      throw new Error(`fixture only has ${corpusTexts.length} passages, need ${N}. Rebuild with --n=${N}.`);
    }
    const docs = corpusTexts.slice(0, N);

    console.log('\n[1] Corpus embedding (batch-64):');
    const embeddings = await embedCorpus(addon, docs, `${N} passages`);

    console.log('\n[2] Building GpuIndex...');
    const tIdx = performance.now();
    const index = new GpuIndex({ docs, embeddings, dim: EMBED_DIM });
    console.log(`  GpuIndex build: ${(performance.now() - tIdx).toFixed(1)}ms (H2D of ${(N * EMBED_DIM * 4 / 1e6).toFixed(1)} MB)`);

    console.log('\n[3] Query latencies (WARM, p50 over 50 iters):');
    for (const { B, label } of QUERY_SHAPES) {
      // Cycle through available queries if B exceeds the pool.
      const qs = Array.from({ length: B }, (_, i) => queryTexts[i % queryTexts.length]);
      await benchQueryShape(addon, index, qs, label);
    }

    index.close();
  }

  console.log('\nDONE');
}

main().catch((e) => { console.error('FAIL:', e); process.exit(1); });
