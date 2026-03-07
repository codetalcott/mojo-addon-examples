const assert = require('node:assert');
const addon = require('./build/search.node');

const buf = Buffer.from('hello world hello');
assert.strictEqual(addon.countByte(buf, 'l'.charCodeAt(0)), 5, 'countByte');
assert.strictEqual(addon.countLines(Buffer.from('a\nb\nc\n')), 3, 'countLines');

const positions = addon.searchAll(buf, Buffer.from('hello'));
assert.strictEqual(positions.length, 2, 'searchAll count');
assert.strictEqual(positions[0], 0, 'searchAll pos 0');
assert.strictEqual(positions[1], 12, 'searchAll pos 1');

console.log('simd-search: OK');
