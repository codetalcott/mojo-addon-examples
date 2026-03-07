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

console.log('stats: OK');
