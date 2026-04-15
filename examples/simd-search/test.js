const assert = require('node:assert');
const addon = require('./build/search.node');

const buf = Buffer.from('hello world hello');
assert.strictEqual(addon.countByte(buf, 'l'.charCodeAt(0)), 5, 'countByte');
assert.strictEqual(addon.countLines(Buffer.from('a\nb\nc\n')), 3, 'countLines');

const positions = addon.searchAll(buf, Buffer.from('hello'));
assert.strictEqual(positions.length, 2, 'searchAll count');
assert.strictEqual(positions[0], 0, 'searchAll pos 0');
assert.strictEqual(positions[1], 12, 'searchAll pos 1');

// Edge cases: empty and single-byte buffers
assert.strictEqual(addon.countByte(Buffer.alloc(0), 0x41), 0, 'countByte empty');
assert.strictEqual(addon.countByte(Buffer.from('A'), 0x41), 1, 'countByte single match');
assert.strictEqual(addon.countByte(Buffer.from('B'), 0x41), 0, 'countByte single no match');
assert.strictEqual(addon.countLines(Buffer.alloc(0)), 0, 'countLines empty');

// Edge cases: searchAll boundaries
const noMatch = addon.searchAll(buf, Buffer.from('xyz'));
assert.strictEqual(noMatch.length, 0, 'searchAll no match');

const tooLong = addon.searchAll(Buffer.from('hi'), Buffer.from('hello'));
assert.strictEqual(tooLong.length, 0, 'searchAll needle longer than buffer');

// Uint8Array input
const u8 = new Uint8Array(Buffer.from('hello world hello'));
const u8Positions = addon.searchAll(Buffer.from(u8), Buffer.from('hello'));
assert.strictEqual(u8Positions.length, 2, 'searchAll Uint8Array');

console.log('simd-search: OK');
