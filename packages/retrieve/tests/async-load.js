// packages/retrieve/tests/async-load.js — concurrency gate G2.
//
// Fires 100 concurrent searches, samples perf_hooks.monitorEventLoopDelay,
// asserts p99 jitter < 5 ms. Also runs the same load through the sync path
// for contrast — the sync version is expected to blow past the threshold
// because every call blocks the JS thread until the GPU round-trips.
//
// Run:  pixi run node packages/retrieve/tests/async-load.js
// Pass: event-loop p99 jitter < 5 ms for the async run
//
// Note: this script requires a GPU and a built packages/retrieve addon. On hosts
// without a GPU it will exit with code 2 (skip) rather than fail.

const path = require('path');
const { monitorEventLoopDelay } = require('perf_hooks');
const { GpuIndex } = require('../lib/GpuIndex');

const DIM = 384;
const N = 10000;
const K = 10;
const CONCURRENCY = 100;
const WARMUP = 8;
const P99_LIMIT_MS = 5;

function makeNormalized(n, dim, seed = 42) {
  // Deterministic Gaussian-ish via LCG + two-sample mean, row-normalized.
  let s = seed >>> 0;
  const next = () => { s = (s * 1664525 + 1013904223) >>> 0; return (s >>> 8) / (1 << 24); };
  const out = new Float32Array(n * dim);
  for (let i = 0; i < n; i++) {
    let norm2 = 0;
    for (let j = 0; j < dim; j++) {
      const v = (next() + next() + next() - 1.5) * 2;
      out[i * dim + j] = v;
      norm2 += v * v;
    }
    const inv = 1 / Math.sqrt(Math.max(norm2, 1e-12));
    for (let j = 0; j < dim; j++) out[i * dim + j] *= inv;
  }
  return out;
}

function summariseHistogram(h) {
  // histogram values are in nanoseconds.
  return {
    minMs: h.min / 1e6,
    meanMs: h.mean / 1e6,
    p50Ms: h.percentile(50) / 1e6,
    p95Ms: h.percentile(95) / 1e6,
    p99Ms: h.percentile(99) / 1e6,
    maxMs: h.max / 1e6,
  };
}

function formatHistogram(label, s) {
  return `${label.padEnd(16)} min=${s.minMs.toFixed(2)}ms mean=${s.meanMs.toFixed(2)}ms p50=${s.p50Ms.toFixed(2)}ms p95=${s.p95Ms.toFixed(2)}ms p99=${s.p99Ms.toFixed(2)}ms max=${s.maxMs.toFixed(2)}ms`;
}

async function runAsync(index, queries) {
  const h = monitorEventLoopDelay({ resolution: 1 });
  h.enable();
  const tStart = performance.now();
  await Promise.all(queries.map((q) => index.searchAsync(q, K)));
  const elapsed = performance.now() - tStart;
  h.disable();
  return { elapsed, hist: summariseHistogram(h) };
}

function runSync(index, queries) {
  const h = monitorEventLoopDelay({ resolution: 1 });
  h.enable();
  const tStart = performance.now();
  for (const q of queries) index.search(q, K);
  const elapsed = performance.now() - tStart;
  h.disable();
  return { elapsed, hist: summariseHistogram(h) };
}

async function main() {
  // Verify GPU is actually available by attempting an index build. If it
  // fails, treat as "skip" rather than "fail" so CI on CPU hosts doesn't
  // flag this test.
  const embeddings = makeNormalized(N, DIM, 1);
  const docs = Array.from({ length: N }, (_, i) => `doc-${i}`);
  let index;
  try {
    index = new GpuIndex({ docs, embeddings, dim: DIM });
  } catch (e) {
    console.error(`[skip] GpuIndex build failed (no GPU?): ${e.message}`);
    process.exit(2);
  }

  const queries = Array.from({ length: CONCURRENCY }, (_, i) =>
    embeddings.subarray((i % N) * DIM, ((i % N) + 1) * DIM).slice(),
  );

  console.log(`corpus: N=${N} dim=${DIM}; concurrency=${CONCURRENCY}; k=${K}`);

  // Warm-up (async path exercises AsyncWork, sync path exercises the sync
  // kernel — both warm the GPU kernel JIT cache).
  console.log('warming up...');
  for (let i = 0; i < WARMUP; i++) {
    await index.searchAsync(queries[0], K);
    index.search(queries[0], K);
  }

  // Measure async path.
  console.log(`\n--- async (${CONCURRENCY} concurrent searchAsync) ---`);
  const asyncResult = await runAsync(index, queries);
  console.log(`wall: ${asyncResult.elapsed.toFixed(1)} ms`);
  console.log(formatHistogram('event-loop:', asyncResult.hist));

  // Measure sync path for contrast.
  console.log(`\n--- sync (${CONCURRENCY} serial search) ---`);
  const syncResult = runSync(index, queries);
  console.log(`wall: ${syncResult.elapsed.toFixed(1)} ms`);
  console.log(formatHistogram('event-loop:', syncResult.hist));

  index.close();

  // Gate G2.
  const pass = asyncResult.hist.p99Ms < P99_LIMIT_MS;
  console.log(
    `\n${pass ? 'GATE G2 PASS' : 'GATE G2 FAIL'}: async event-loop p99 = ${asyncResult.hist.p99Ms.toFixed(2)} ms (limit ${P99_LIMIT_MS} ms)`,
  );
  process.exit(pass ? 0 : 1);
}

main().catch((e) => {
  console.error('FAIL:', e);
  process.exit(1);
});
