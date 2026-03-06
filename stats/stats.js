// stats/stats.js — SIMD statistics benchmark
//
// Demonstrates Mojo's SIMD reductions (sum/min/max/variance) vs pure JS.
// stats() returns {mean, stddev, min, max, p50, p95, p99} in a single call.
//
// Build:  pixi run bash stats/build.sh
// Run:    node stats/stats.js

const addon = require('./build/stats.node');

// --- Pure JS baseline --------------------------------------------------------

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

  // Sort a copy for percentiles
  const sorted = Float64Array.from(data).sort();
  const p50 = sorted[Math.floor((data.length - 1) * 0.5)];
  const p95 = sorted[Math.floor((data.length - 1) * 0.95)];
  const p99 = sorted[Math.floor((data.length - 1) * 0.99)];

  return { mean, stddev, min, max, p50, p95, p99 };
}

function jsHistogram(data, bins) {
  let min = data[0], max = data[0];
  for (let i = 0; i < data.length; i++) {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }
  const counts = new Float64Array(bins);
  const range = max - min;
  if (range === 0) { counts[0] = data.length; return counts; }
  for (let i = 0; i < data.length; i++) {
    let idx = Math.floor((data[i] - min) / range * bins);
    if (idx >= bins) idx = bins - 1;
    counts[idx]++;
  }
  return counts;
}

// --- Correctness checks ------------------------------------------------------

console.log('=== Correctness ===\n');

const testData = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const jsResult = jsStats(testData);
const mojoResult = addon.stats(testData);

function approxEq(a, b, eps = 1e-10) { return Math.abs(a - b) < eps; }

const checks = [
  ['mean', 5.5],
  ['min', 1],
  ['max', 10],
  ['p50', 5],
  ['p95', 9],
  ['p99', 9],
];

for (const [key, expected] of checks) {
  const val = mojoResult[key];
  const pass = approxEq(val, expected, 0.5);
  console.log(`  ${key}: ${val.toFixed(4)} (expected ~${expected}) ${pass ? 'PASS' : 'FAIL'}`);
}

// Stddev check (population stddev of 1..10 = 2.8722...)
const stddevPass = approxEq(mojoResult.stddev, jsResult.stddev, 0.01);
console.log(`  stddev: ${mojoResult.stddev.toFixed(4)} (expected ~${jsResult.stddev.toFixed(4)}) ${stddevPass ? 'PASS' : 'FAIL'}`);

// Histogram check
const histData = new Float64Array([0, 0.25, 0.5, 0.75, 1.0]);
const hist = addon.histogram(histData, 4);
const histSum = Array.from(hist).reduce((a, b) => a + b, 0);
console.log(`  histogram: [${Array.from(hist)}] sum=${histSum} (expected 5) ${histSum === 5 ? 'PASS' : 'FAIL'}`);

// --- Benchmark ---------------------------------------------------------------

function bench(name, fn, iters) {
  for (let i = 0; i < Math.min(iters, 20); i++) fn(); // warmup
  const start = performance.now();
  for (let i = 0; i < iters; i++) fn();
  const ms = performance.now() - start;
  const opsPerSec = Math.round(iters / (ms / 1000));
  return { name, ms, opsPerSec };
}

function formatOps(ops) {
  if (ops >= 1e6) return (ops / 1e6).toFixed(1) + 'M';
  if (ops >= 1e3) return (ops / 1e3).toFixed(1) + 'K';
  return ops.toString();
}

const sizes = [
  [10_000, 10000],
  [100_000, 1000],
  [1_000_000, 100],
  [10_000_000, 10],
];

for (const [SIZE, ITERS] of sizes) {
  console.log(`\n=== stats(): ${SIZE.toLocaleString()} elements ===\n`);

  const data = new Float64Array(SIZE);
  for (let i = 0; i < SIZE; i++) data[i] = Math.random() * 1000;

  const jsR = bench('JS       ', () => jsStats(data), ITERS);
  const mojoR = bench('Mojo SIMD', () => addon.stats(data), ITERS);

  const speedup = (jsR.ms / mojoR.ms).toFixed(1);
  console.log(`  ${jsR.name}: ${jsR.ms.toFixed(1)}ms  ${formatOps(jsR.opsPerSec)} ops/sec  (baseline)`);
  console.log(`  ${mojoR.name}: ${mojoR.ms.toFixed(1)}ms  ${formatOps(mojoR.opsPerSec)} ops/sec  ${speedup}x`);
}

// Histogram benchmark
console.log('\n--- histogram benchmark ---');

for (const [SIZE, ITERS] of [[1_000_000, 100], [10_000_000, 10]]) {
  console.log(`\n=== histogram(): ${SIZE.toLocaleString()} elements, 100 bins ===\n`);

  const data = new Float64Array(SIZE);
  for (let i = 0; i < SIZE; i++) data[i] = Math.random() * 1000;

  const jsR = bench('JS       ', () => jsHistogram(data, 100), ITERS);
  const mojoR = bench('Mojo SIMD', () => addon.histogram(data, 100), ITERS);

  const speedup = (jsR.ms / mojoR.ms).toFixed(1);
  console.log(`  ${jsR.name}: ${jsR.ms.toFixed(1)}ms  ${formatOps(jsR.opsPerSec)} ops/sec  (baseline)`);
  console.log(`  ${mojoR.name}: ${mojoR.ms.toFixed(1)}ms  ${formatOps(mojoR.opsPerSec)} ops/sec  ${speedup}x`);
}
