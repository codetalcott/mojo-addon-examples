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


// Parallel path: _count_byte splits across workers above PARALLEL_THRESHOLD
// (64KB). Everything above stays on the sequential path.

// Deterministic PRNG so failures reproduce exactly.
function lcg(seed) {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return s / 4294967296; };
}

{
  const rnd = lcg(99);
  const N = 300000; // comfortably over the 64KB threshold
  const big = Buffer.alloc(N);
  for (let i = 0; i < N; i++) big[i] = Math.floor(rnd() * 256);
  for (const target of [0x00, 0x0a, 0x41, 0xff]) {
    let expect = 0;
    for (let i = 0; i < N; i++) if (big[i] === target) expect++;
    assert.strictEqual(addon.countByte(big, target), expect, `countByte parallel 0x${target.toString(16)}`);
  }
  let lines = 0;
  for (let i = 0; i < N; i++) if (big[i] === 0x0a) lines++;
  assert.strictEqual(addon.countLines(big), lines, 'countLines parallel');

  // searchAll over the same buffer: single-byte needle uses the parallel count
  // to size its result array, so a bad count truncates the positions.
  const found = addon.searchAll(big, Buffer.from([0x41]));
  const expectPos = [];
  for (let i = 0; i < N; i++) if (big[i] === 0x41) expectPos.push(i);
  assert.strictEqual(found.length, expectPos.length, 'searchAll parallel count');
  assert.deepStrictEqual(Array.from(found), expectPos, 'searchAll parallel positions');
}

console.log('simd-search: OK');
