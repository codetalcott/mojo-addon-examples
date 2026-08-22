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

// --- Parallel-path coverage (size >= PARALLEL_THRESHOLD) ----------------------
// Everything above is at most 17 bytes, so `_count_byte` returns from its
// `if size < PARALLEL_THRESHOLD` (65536) serial branch and the four-worker split
// never executes. countByte and countLines both route through `_count_byte`, as
// does searchAll — but only for a *single-byte* needle; a longer needle is
// counted by `_count_multi_byte`, which has no parallel path at all.
//
// Two sizes: one an exact multiple of NUM_WORKERS (4), one not, so the last
// worker's `else size` remainder branch runs too.

const NEEDLE = Buffer.from('QZ7');

function buildHaystack(size) {
  const buf = Buffer.alloc(size, 0x61); // 'a'
  // Sprinkle deterministic markers: newlines, a target byte, and the needle.
  for (let i = 0; i < size; i++) {
    if (i % 97 === 0) buf[i] = 0x0a;       // '\n'
    else if (i % 31 === 0) buf[i] = 0x7a;  // 'z'
  }
  // Place the needle at positions spread across all four worker chunks,
  // including one straddling a chunk boundary so a mis-split loses it.
  const chunk = Math.floor(size / 4);
  const spots = [
    5,
    chunk + 11,
    chunk - 1,          // straddles the worker 0 / worker 1 boundary
    2 * chunk + 7,
    3 * chunk + 13,
    size - NEEDLE.length - 1,
  ];
  for (const at of spots) NEEDLE.copy(buf, at);
  return buf;
}

function countByteRef(buf, byte) {
  let n = 0;
  for (let i = 0; i < buf.length; i++) if (buf[i] === byte) n++;
  return n;
}

function searchAllRef(hay, needle) {
  const out = [];
  let from = 0;
  for (;;) {
    const at = hay.indexOf(needle, from);
    if (at === -1) break;
    out.push(at);
    from = at + 1; // allow overlapping matches; NEEDLE cannot overlap itself
  }
  return out;
}

for (const size of [131072, 100003]) {
  const hay = buildHaystack(size);

  assert.strictEqual(
    addon.countByte(hay, 0x7a), countByteRef(hay, 0x7a),
    `parallel countByte 'z' (n=${size})`
  );
  assert.strictEqual(
    addon.countByte(hay, 0x61), countByteRef(hay, 0x61),
    `parallel countByte 'a' (n=${size})`
  );
  // A byte that appears nowhere must still come back 0 from every worker.
  assert.strictEqual(
    addon.countByte(hay, 0x00), 0, `parallel countByte absent (n=${size})`
  );
  assert.strictEqual(
    addon.countLines(hay), countByteRef(hay, 0x0a),
    `parallel countLines (n=${size})`
  );

  // Single-byte needle: this is the searchAll shape that reaches the parallel
  // `_count_byte`. The count sizes the result buffer that the (serial) collect
  // pass then fills, so a bad worker split does not just miscount — it
  // under-allocates and the collect writes past the end.
  const zBytes = Array.from(addon.searchAll(hay, Buffer.from([0x7a])));
  const zExpected = [];
  for (let i = 0; i < hay.length; i++) if (hay[i] === 0x7a) zExpected.push(i);
  assert.deepStrictEqual(
    zBytes, zExpected, `parallel searchAll single-byte needle (n=${size})`
  );

  // Multi-byte needle takes the serial `_count_multi_byte` path; included for
  // large-input coverage of its SIMD-chunk/scalar-tail split, not the workers.
  const expected = searchAllRef(hay, NEEDLE);
  assert(expected.length >= 5, `fixture places multiple needles (n=${size})`);
  const positions = Array.from(addon.searchAll(hay, NEEDLE));
  assert.deepStrictEqual(
    positions.slice().sort((a, b) => a - b), expected,
    `searchAll multi-byte positions (n=${size})`
  );
}

console.log('simd-search: OK');
