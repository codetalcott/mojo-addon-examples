// Regression test for the Phase 3a.1 persistent-device-buffer prototype.
//
// Verifies that loadGpu / countByteHandle / releaseGpu produce byte-exact
// counts matching a plain JS scan, across a range of sizes and target bytes.
// Also runs a tight load-release loop as a basic resource leak smoke test.
//
// Run: node simd-search/test_cached.js

const assert = require('node:assert');

let cached;
try {
  cached = require('./build/search_cached.node');
} catch (e) {
  console.error('simd-search-cached: build not found. Run: pixi run bash simd-search/build_cached.sh');
  process.exit(1);
}

function jsCountByte(buf, byte) {
  let n = 0;
  for (let i = 0; i < buf.length; i++) if (buf[i] === byte) n++;
  return n;
}

function makeBuf(size, seed) {
  // Deterministic pseudo-random content so counts are non-trivial but reproducible.
  const buf = Buffer.alloc(size);
  let s = seed | 0;
  for (let i = 0; i < size; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    buf[i] = s & 0xff;
  }
  return buf;
}

// --- Correctness: 5 sizes x 5 target bytes ---------------------------------

const sizes = [
  0,             // edge: empty
  1,             // edge: single byte
  257,           // smaller than one GPU block chunk
  65537,         // straddles a block boundary
  3_145_729,     // ~3 MB, many blocks
];
const targets = [0x00, 0x41, 0x7f, 0xaa, 0xff];

let totalCases = 0;
for (const size of sizes) {
  const buf = makeBuf(size, 0xC0FFEE + size);
  // Empty buffer is a valid input but produces a zero-length allocation;
  // skip it for the handle path (loadGpu on size=0 is undefined behavior).
  if (size === 0) {
    assert.strictEqual(jsCountByte(buf, 0x41), 0, 'js sanity: empty');
    continue;
  }
  const h = cached.loadGpu(buf);
  for (const tgt of targets) {
    const expected = jsCountByte(buf, tgt);
    const got = cached.countByteHandle(h, tgt);
    assert.strictEqual(
      got,
      expected,
      `size=${size} target=0x${tgt.toString(16)} expected=${expected} got=${got}`,
    );
    totalCases++;
  }
  // After release, further queries must throw.
  cached.releaseGpu(h);
  let threw = false;
  try {
    cached.countByteHandle(h, 0x41);
  } catch (e) {
    threw = true;
  }
  assert.ok(threw, `size=${size}: countByteHandle after releaseGpu should throw`);
}

// --- Stability: repeated queries against the same handle -------------------
// Ensures per-call state (partial sums buffer, etc.) is reused correctly and
// doesn't accumulate or corrupt across calls.

{
  const buf = makeBuf(1_048_576, 0xBEEF);
  const expected = jsCountByte(buf, 0x41);
  const h = cached.loadGpu(buf);
  for (let i = 0; i < 200; i++) {
    const got = cached.countByteHandle(h, 0x41);
    assert.strictEqual(got, expected, `repeated query iter ${i}`);
  }
  cached.releaseGpu(h);
  totalCases += 200;
}

// --- Leak smoke test: load + release 500 times, check RSS stability --------
// We don't assert a bound — just log before/after so a human can spot runaway
// growth. GPU memory is released lazily via GC, so we call global.gc() if
// available (run with --expose-gc to enable).

{
  const rssBefore = process.memoryUsage.rss();
  const buf = makeBuf(1_048_576, 0xDEADBEEF);
  for (let i = 0; i < 500; i++) {
    const h = cached.loadGpu(buf);
    cached.countByteHandle(h, i & 0xff);
    cached.releaseGpu(h);
  }
  if (typeof global.gc === 'function') global.gc();
  const rssAfter = process.memoryUsage.rss();
  const deltaMB = (rssAfter - rssBefore) / (1024 * 1024);
  console.log(
    `  leak-smoke: RSS before=${(rssBefore / 1024 / 1024).toFixed(1)}MB ` +
      `after=${(rssAfter / 1024 / 1024).toFixed(1)}MB delta=${deltaMB.toFixed(1)}MB ` +
      `(500 load+release on 1MB; run with --expose-gc for eager reclaim)`,
  );
}

console.log(`simd-search-cached: OK (${totalCases} correctness cases)`);
