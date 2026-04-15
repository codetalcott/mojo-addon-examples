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

console.log('stats: OK');
