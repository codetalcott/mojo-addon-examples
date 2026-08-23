const assert = require('node:assert');
const addon = require('./build/stats.node');

const data = new Float64Array([1, 2, 3, 4, 5]);
const result = addon.stats(data);

assert.strictEqual(result.mean, 3, 'mean');
assert.strictEqual(result.min, 1, 'min');
assert.strictEqual(result.max, 5, 'max');
assert(Math.abs(result.stddev - Math.sqrt(2)) < 0.001, 'stddev');

const hist = addon.histogram(data, 2);
assert.strictEqual(hist.length, 2, 'histogram bins');

// Single element: mean=value, stddev=0, min=max=value
const single = addon.stats(new Float64Array([42]));
assert.strictEqual(single.mean, 42, 'single mean');
assert.strictEqual(single.min, 42, 'single min');
assert.strictEqual(single.max, 42, 'single max');
assert.strictEqual(single.stddev, 0, 'single stddev');

// All identical values: stddev=0
const same = addon.stats(new Float64Array([7, 7, 7, 7]));
assert.strictEqual(same.stddev, 0, 'identical stddev');
assert.strictEqual(same.mean, 7, 'identical mean');

// Two elements: verifiable percentiles
const pair = addon.stats(new Float64Array([10, 20]));
assert.strictEqual(pair.mean, 15, 'pair mean');
assert.strictEqual(pair.min, 10, 'pair min');
assert.strictEqual(pair.max, 20, 'pair max');

// --- Parallel-path coverage (size >= PARALLEL_THRESHOLD) ----------------------
// Every case above is at most 5 elements, so `_parallel_sum_min_max` and
// `_parallel_sum_sq_diff` short-circuit to their serial branch at
// `if size < PARALLEL_THRESHOLD` (4096) and the four-worker split never runs.
// These arrays cross that threshold, so the worker chunk arithmetic is actually
// executed and a bad split shows up as a wrong sum, min, or max.
//
// Two sizes: one an exact multiple of NUM_WORKERS (4), one not, so the
// `wid < NUM_WORKERS - 1 else size` branch where the last worker takes the
// remainder is covered as well.

function statsRef(arr) {
  let sum = 0, min = arr[0], max = arr[0];
  for (const v of arr) {
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const mean = sum / arr.length;
  let sumSq = 0;
  for (const v of arr) sumSq += (v - mean) * (v - mean);
  return { mean, min, max, stddev: Math.sqrt(sumSq / arr.length) };
}

// Relative tolerance: the kernel accumulates across SIMD lanes and four worker
// partials, so its summation order differs from the reference loop. Values
// agree to well within this; min/max are compared exactly since neither rounds.
function assertClose(actual, expected, label) {
  const tol = Math.max(1e-9, Math.abs(expected) * 1e-9);
  assert(
    Math.abs(actual - expected) <= tol,
    `${label}: expected ${expected}, got ${actual}`
  );
}

for (const size of [8192, 5001]) {
  // Deterministic, non-uniform, and spread across positive and negative values
  // so a dropped chunk moves the sum, and min/max sit at known interior
  // positions rather than at the array edges where an off-by-one would hide.
  const arr = new Float64Array(size);
  let seed = 0x51ed270b;
  for (let i = 0; i < size; i++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    arr[i] = ((seed >> 8) % 20000) / 100 - 100; // ~[-100, 100)
  }
  const midLow = Math.floor(size / 2) + 1;
  const midHigh = Math.floor(size / 3);
  arr[midLow] = -1234.5;  // unique min, interior
  arr[midHigh] = 4321.5;  // unique max, interior

  const ref = statsRef(arr);
  const got = addon.stats(arr);

  assertClose(got.mean, ref.mean, `parallel mean (n=${size})`);
  assertClose(got.stddev, ref.stddev, `parallel stddev (n=${size})`);
  assert.strictEqual(got.min, ref.min, `parallel min (n=${size})`);
  assert.strictEqual(got.max, ref.max, `parallel max (n=${size})`);
  assert.strictEqual(got.min, -1234.5, `parallel min is the interior value (n=${size})`);
  assert.strictEqual(got.max, 4321.5, `parallel max is the interior value (n=${size})`);

  // Percentiles come from quickselect, not the parallel split; just require
  // them ordered and inside the observed range.
  assert(got.p50 <= got.p95 && got.p95 <= got.p99, `percentiles ordered (n=${size})`);
  assert(got.p50 >= ref.min && got.p99 <= ref.max, `percentiles in range (n=${size})`);

  // histogram() reaches the same parallel min/max helper; every element must
  // land in exactly one bin.
  const hist = addon.histogram(arr, 16);
  assert.strictEqual(hist.length, 16, `parallel histogram bins (n=${size})`);
  const binned = Array.from(hist).reduce((a, b) => a + b, 0);
  assert.strictEqual(binned, size, `parallel histogram total (n=${size})`);
}

console.log('stats: OK');
