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


// Parallel path: _parallel_sum_min_max / _parallel_sum_sq_diff split across
// workers at PARALLEL_THRESHOLD (4096). The cases above are all below it.

// Deterministic PRNG so failures reproduce exactly.
function lcg(seed) {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return s / 4294967296; };
}

{
  const rnd = lcg(12345);
  for (const N of [4096, 20000]) {
    const d = Float64Array.from({ length: N }, () => rnd() * 1000);
    const r = addon.stats(d);
    let sum = 0, mn = Infinity, mx = -Infinity;
    for (let i = 0; i < N; i++) { sum += d[i]; if (d[i] < mn) mn = d[i]; if (d[i] > mx) mx = d[i]; }
    const mean = sum / N;
    let sq = 0;
    for (let i = 0; i < N; i++) { const t = d[i] - mean; sq += t * t; }
    const sorted = Array.from(d).sort((a, b) => a - b);
    const pct = (q) => sorted[Math.floor((N - 1) * q)];
    assert(Math.abs(r.mean - mean) < 1e-9, `parallel mean N=${N}`);
    assert(Math.abs(r.stddev - Math.sqrt(sq / N)) < 1e-9, `parallel stddev N=${N}`);
    assert.strictEqual(r.min, mn, `parallel min N=${N}`);
    assert.strictEqual(r.max, mx, `parallel max N=${N}`);
    assert(Math.abs(r.p50 - pct(0.5)) < 1e-9, `parallel p50 N=${N}`);
    assert(Math.abs(r.p99 - pct(0.99)) < 1e-9, `parallel p99 N=${N}`);

    const h = addon.histogram(d, 10);
    let total = 0;
    for (let i = 0; i < h.length; i++) total += h[i];
    assert.strictEqual(total, N, `parallel histogram total N=${N}`);
  }
}

console.log('stats: OK');
