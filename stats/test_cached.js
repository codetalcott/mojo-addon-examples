// Regression test for the Phase 3b.2 persistent-device-buffer stats addon.
//
// Verifies loadStatsGpu / statsHandle / releaseStatsGpu produce results that
// match the validated CPU path (addon.stats) within Float32 precision tolerance,
// across a range of sizes and seeds. Also runs stability (200 repeat queries
// on one handle) and a leak smoke test.
//
// Run: node stats/test_cached.js
//      node --expose-gc stats/test_cached.js

const assert = require('node:assert');

let addon;
try {
  addon = require('./build/stats.node');
} catch (e) {
  console.error('stats.node not found — run: pixi run bash stats/build.sh');
  process.exit(1);
}

let cached;
try {
  cached = require('./build/stats_cached.node');
} catch (e) {
  console.error('stats_cached.node not found — run: pixi run bash stats/build_cached.sh');
  process.exit(1);
}

function makeData(size, seed) {
  const data = new Float64Array(size);
  let s = seed | 0;
  for (let i = 0; i < size; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    data[i] = (s / 0x7fffffff) * 1000.0;
  }
  return data;
}

// Float32 cast + reduction in Float32 loses ~7 decimal digits. Tolerate
// 1e-4 relative error on mean/stddev; min/max/percentiles should be
// Float32-exact (since they're pass-through comparisons with only one cast).
function closeRel(a, b, tol) {
  if (a === 0 && b === 0) return true;
  const denom = Math.max(Math.abs(a), Math.abs(b));
  return Math.abs(a - b) / denom < tol;
}

// --- Correctness: 4 sizes × 2 seeds = 8 cases ------------------------------

const sizes = [
  1_000,
  10_000,
  100_000,
  1_000_000,
];
const seeds = [0xC0FFEE, 0xBADF00D];

let totalCases = 0;
for (const size of sizes) {
  for (const seed of seeds) {
    const data = makeData(size, seed);
    const expected = addon.stats(data);
    const handle = cached.loadStatsGpu(data);
    const got = cached.statsHandle(handle);
    for (const key of ['mean', 'stddev', 'min', 'max', 'p50', 'p95', 'p99']) {
      assert.ok(
        closeRel(got[key], expected[key], 1e-4),
        `size=${size} seed=0x${seed.toString(16)} ${key}: ` +
          `expected=${expected[key]} got=${got[key]}`,
      );
    }
    cached.releaseStatsGpu(handle);
    let threw = false;
    try {
      cached.statsHandle(handle);
    } catch (e) {
      threw = true;
    }
    assert.ok(threw, `size=${size}: statsHandle after release should throw`);
    totalCases++;
  }
}

// --- Stability: 200 repeated queries on one handle -------------------------
{
  const data = makeData(100_000, 0xBEEF);
  const expected = addon.stats(data);
  const handle = cached.loadStatsGpu(data);
  let firstMean = null;
  for (let i = 0; i < 200; i++) {
    const got = cached.statsHandle(handle);
    if (i === 0) firstMean = got.mean;
    assert.strictEqual(got.mean, firstMean, `stability iter ${i}: mean drifted`);
    if (i === 0 || i === 199) {
      assert.ok(
        closeRel(got.mean, expected.mean, 1e-4),
        `stability iter ${i}: mean=${got.mean} expected=${expected.mean}`,
      );
    }
  }
  cached.releaseStatsGpu(handle);
  totalCases += 200;
}

// --- Leak smoke: 300 load + release iterations on 100K elements -----------
// Strategy doc notes an M4 Metal leak quirk in Phase 3a/3b.1. Log only.
{
  const data = makeData(100_000, 0xDEADBEEF);
  const rssBefore = process.memoryUsage.rss();
  for (let i = 0; i < 300; i++) {
    const handle = cached.loadStatsGpu(data);
    cached.statsHandle(handle);
    cached.releaseStatsGpu(handle);
    if (i % 50 === 49 && typeof global.gc === 'function') global.gc();
  }
  if (typeof global.gc === 'function') global.gc();
  const rssAfter = process.memoryUsage.rss();
  const deltaMB = (rssAfter - rssBefore) / (1024 * 1024);
  console.log(
    `  leak-smoke: RSS before=${(rssBefore / 1024 / 1024).toFixed(1)}MB ` +
      `after=${(rssAfter / 1024 / 1024).toFixed(1)}MB delta=${deltaMB.toFixed(1)}MB ` +
      `(300 load+release on 100K; known M4 Metal quirk — run with --expose-gc for eager reclaim)`,
  );
}

console.log(`stats-cached: OK (${totalCases} correctness cases)`);
