// stats/stats_cached.js — Phase 3b.2 persistent device buffer stats benchmark
//
// Four-path comparison for stats() at each size:
//   1. JS loop              — baseline (sum/min/max/variance + sort for percentiles)
//   2. Mojo CPU SIMD        — existing stats.node stats()
//   3. Mojo GPU one-shot    — existing stats.node statsGpu() (re-cast + H2D per call)
//   4. Mojo GPU cached      — new stats_cached.node, loadStatsGpu once, statsHandle per call
//
// The cached path amortizes the Float64→Float32 cast + H2D transfer + device
// allocation across all subsequent statsHandle calls. Per-call cost is two
// kernel launches + CPU-side quickselect on a scratch copy of the cached
// Float64 data for percentiles. Expected H100 win: the 5-10× gap closed in
// Phase 2d becomes 30-100× cached.
//
// Build:  pixi run bash stats/build.sh && pixi run bash stats/build_cached.sh
// Run:    node stats/stats_cached.js

const addon = require('./build/stats.node');

let cached = null;
try {
  cached = require('./build/stats_cached.node');
} catch (e) {
  console.error('stats_cached.node not found — run: pixi run bash stats/build_cached.sh');
  process.exit(1);
}

// --- JS baseline -------------------------------------------------------------

function jsStats(data) {
  let sum = 0, min = data[0], max = data[0];
  for (let i = 0; i < data.length; i++) {
    sum += data[i];
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }
  const mean = sum / data.length;

  let sumSq = 0;
  for (let i = 0; i < data.length; i++) {
    const d = data[i] - mean;
    sumSq += d * d;
  }
  const stddev = Math.sqrt(sumSq / data.length);

  const sorted = Float64Array.from(data).sort();
  const p50 = sorted[Math.floor((data.length - 1) * 0.5)];
  const p95 = sorted[Math.floor((data.length - 1) * 0.95)];
  const p99 = sorted[Math.floor((data.length - 1) * 0.99)];

  return { mean, stddev, min, max, p50, p95, p99 };
}

// --- Bench primitives --------------------------------------------------------

function bench(name, fn, iters) {
  for (let i = 0; i < Math.min(iters, 5); i++) fn();
  const start = performance.now();
  for (let i = 0; i < iters; i++) fn();
  const ms = performance.now() - start;
  const msPerOp = ms / iters;
  return { name, ms, msPerOp };
}

function formatCount(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(0) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(0) + 'K';
  return n.toString();
}

// --- Correctness spot-check --------------------------------------------------

{
  const testData = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  const jsR = jsStats(testData);
  const cpuR = addon.stats(testData);
  const handle = cached.loadStatsGpu(testData);
  const cachedR = cached.statsHandle(handle);
  cached.releaseStatsGpu(handle);

  const closeRel = (a, b, tol = 1e-4) => {
    if (a === 0 && b === 0) return true;
    const d = Math.max(Math.abs(a), Math.abs(b));
    return Math.abs(a - b) / d < tol;
  };

  console.log('=== Correctness ===\n');
  let ok = true;
  for (const key of ['mean', 'stddev', 'min', 'max', 'p50', 'p95', 'p99']) {
    const match = closeRel(cachedR[key], jsR[key]);
    if (!match) ok = false;
    console.log(`  ${key}: js=${jsR[key].toFixed(4)} cpu=${cpuR[key].toFixed(4)} cached=${cachedR[key].toFixed(4)} ${match ? 'PASS' : 'FAIL'}`);
  }
  if (!ok) {
    console.error('  FAIL: mismatch');
    process.exit(1);
  }
  console.log('  PASS\n');
}

// --- Main benchmark loop -----------------------------------------------------

const sizes = [
  [100_000,    200],
  [1_000_000,  100],
  [10_000_000,  20],
];

console.log('--- stats benchmark (4 paths) ---\n');
console.log('  cached row reports per-call cost *after* loadStatsGpu.\n');

for (const [SIZE, ITERS] of sizes) {
  console.log(`=== stats: ${formatCount(SIZE)} elements (${ITERS} iters) ===\n`);

  const data = new Float64Array(SIZE);
  for (let i = 0; i < SIZE; i++) data[i] = Math.random() * 1000;

  // 1. JS baseline
  const jsR = bench('JS       ', () => jsStats(data), ITERS);
  console.log(`  ${jsR.name}: ${jsR.msPerOp.toFixed(2)}ms/op  (baseline)`);

  // 2. Mojo CPU SIMD
  const cpuR = bench('Mojo SIMD', () => addon.stats(data), ITERS);
  const cpuSpeed = (jsR.msPerOp / cpuR.msPerOp).toFixed(1);
  console.log(`  ${cpuR.name}: ${cpuR.msPerOp.toFixed(2)}ms/op  ${cpuSpeed}x`);

  // 3. Mojo GPU one-shot
  let gpuR = null;
  try {
    // Spot-check Float32 precision before benchmarking.
    const gpuSpot = addon.statsGpu(data);
    const cpuSpot = addon.stats(data);
    const tol = 1e-4;
    for (const k of ['mean', 'stddev', 'min', 'max']) {
      if (Math.abs(gpuSpot[k] - cpuSpot[k]) / Math.max(Math.abs(cpuSpot[k]), 1e-12) > tol) {
        throw new Error(`${k} mismatch: gpu=${gpuSpot[k]} cpu=${cpuSpot[k]}`);
      }
    }
    gpuR = bench('Mojo GPU ', () => addon.statsGpu(data), ITERS);
    const gSpeed = (jsR.msPerOp / gpuR.msPerOp).toFixed(1);
    console.log(`  ${gpuR.name}: ${gpuR.msPerOp.toFixed(2)}ms/op  ${gSpeed}x`);
  } catch (e) {
    console.log(`  Mojo GPU : skipped (${e.message})`);
  }

  // 4. Mojo GPU cached
  try {
    const loadStart = performance.now();
    const handle = cached.loadStatsGpu(data);
    const loadMs = performance.now() - loadStart;

    // Spot-check correctness before benchmarking.
    const cachedSpot = cached.statsHandle(handle);
    const cpuSpot = addon.stats(data);
    const tol = 1e-4;
    for (const k of ['mean', 'stddev', 'min', 'max']) {
      if (Math.abs(cachedSpot[k] - cpuSpot[k]) / Math.max(Math.abs(cpuSpot[k]), 1e-12) > tol) {
        cached.releaseStatsGpu(handle);
        throw new Error(`${k} mismatch: cached=${cachedSpot[k]} cpu=${cpuSpot[k]}`);
      }
    }

    const cachedR = bench('GPU cache', () => cached.statsHandle(handle), ITERS);
    cached.releaseStatsGpu(handle);

    const cSpeed = (jsR.msPerOp / cachedR.msPerOp).toFixed(1);
    let suffix = `   (loadStatsGpu: ${loadMs.toFixed(1)}ms one-time)`;
    if (gpuR && gpuR.msPerOp > cachedR.msPerOp) {
      const perCallSavings = gpuR.msPerOp - cachedR.msPerOp;
      const breakEven = Math.ceil(loadMs / perCallSavings);
      suffix += `   break-even vs one-shot: ${breakEven} iters`;
    }
    console.log(`  ${cachedR.name}: ${cachedR.msPerOp.toFixed(2)}ms/op  ${cSpeed}x${suffix}`);
  } catch (e) {
    console.log(`  GPU cache: skipped (${e.message})`);
  }

  console.log('');
}
