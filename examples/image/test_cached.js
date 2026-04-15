// Regression test for the Phase 3b.1 persistent-device-buffer grayscale addon.
//
// Verifies that loadImageGpu / grayscaleHandle / releaseImageGpu produce
// byte-exact output matching the validated CPU SIMD path (addon.grayscale),
// across a range of sizes and seeds. Also runs stability (200 repeat queries
// on one handle) and a leak smoke test (500 load+release iterations).
//
// Run: node image/test_cached.js
//      node --expose-gc image/test_cached.js   (for eager GC during leak smoke)

const assert = require('node:assert');

let addon;
try {
  addon = require('./build/image.node');
} catch (e) {
  console.error('image.node not found — run: pixi run bash image/build.sh');
  process.exit(1);
}

let cached;
try {
  cached = require('./build/image_cached.node');
} catch (e) {
  console.error('image_cached.node not found — run: pixi run bash image/build_cached.sh');
  process.exit(1);
}

function makeRgba(w, h, seed) {
  // Deterministic LCG over all four channels (including alpha, so the
  // kernel's alpha-preserve path is exercised on non-trivial data).
  const rgba = new Uint8Array(w * h * 4);
  let s = seed | 0;
  for (let i = 0; i < rgba.length; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    rgba[i] = s & 0xff;
  }
  return rgba;
}

// --- Correctness: 3 sizes × 3 seeds = 9 cases ------------------------------

const sizes = [
  [100, 100],       // small: single-block grid
  [720, 480],       // SD
  [1920, 1080],     // FHD: ~8.3 MB
];
const seeds = [0xC0FFEE, 0xBADF00D, 0x1337];

let totalCases = 0;
for (const [w, h] of sizes) {
  for (const seed of seeds) {
    const src = makeRgba(w, h, seed);
    const expected = addon.grayscale(src, w, h); // CPU SIMD oracle
    const handle = cached.loadImageGpu(src, w, h);
    const dst = new Uint8Array(w * h * 4);
    cached.grayscaleHandle(handle, dst);
    assert.deepStrictEqual(
      Buffer.from(dst.buffer, dst.byteOffset, dst.byteLength),
      Buffer.from(expected.buffer, expected.byteOffset, expected.byteLength),
      `mismatch at ${w}x${h} seed=0x${seed.toString(16)}`,
    );
    cached.releaseImageGpu(handle);
    let threw = false;
    try {
      cached.grayscaleHandle(handle, dst);
    } catch (e) {
      threw = true;
    }
    assert.ok(threw, `${w}x${h}: grayscaleHandle after release should throw`);
    totalCases++;
  }
}

// --- dst too small should throw --------------------------------------------
{
  const w = 64, h = 64;
  const src = makeRgba(w, h, 0x5EED);
  const handle = cached.loadImageGpu(src, w, h);
  const tooSmall = new Uint8Array(w * h * 4 - 1);
  let threw = false;
  try {
    cached.grayscaleHandle(handle, tooSmall);
  } catch (e) {
    threw = true;
  }
  assert.ok(threw, 'short dst buffer should throw');
  cached.releaseImageGpu(handle);
  totalCases++;
}

// --- Stability: 200 repeated queries on one handle -------------------------
{
  const w = 1280, h = 720;
  const src = makeRgba(w, h, 0xBEEF);
  const expected = addon.grayscale(src, w, h);
  const handle = cached.loadImageGpu(src, w, h);
  const dst = new Uint8Array(w * h * 4);
  for (let i = 0; i < 200; i++) {
    cached.grayscaleHandle(handle, dst);
    if (i === 0 || i === 199) {
      assert.deepStrictEqual(
        Buffer.from(dst.buffer, dst.byteOffset, dst.byteLength),
        Buffer.from(expected.buffer, expected.byteOffset, expected.byteLength),
        `stability iter ${i}`,
      );
    }
  }
  cached.releaseImageGpu(handle);
  totalCases += 200;
}

// --- Leak smoke: 500 load + release iterations on 720p --------------------
// Strategy doc notes an M4 Metal leak quirk in Phase 3a; this test *logs*
// the RSS delta but does not fail on growth. Root cause investigation is
// deferred to Phase 3b.3 Track A.
{
  const w = 1280, h = 720;
  const src = makeRgba(w, h, 0xDEADBEEF);
  const dst = new Uint8Array(w * h * 4);
  const rssBefore = process.memoryUsage.rss();
  for (let i = 0; i < 500; i++) {
    const handle = cached.loadImageGpu(src, w, h);
    cached.grayscaleHandle(handle, dst);
    cached.releaseImageGpu(handle);
    if (i % 50 === 49 && typeof global.gc === 'function') global.gc();
  }
  if (typeof global.gc === 'function') global.gc();
  const rssAfter = process.memoryUsage.rss();
  const deltaMB = (rssAfter - rssBefore) / (1024 * 1024);
  console.log(
    `  leak-smoke: RSS before=${(rssBefore / 1024 / 1024).toFixed(1)}MB ` +
      `after=${(rssAfter / 1024 / 1024).toFixed(1)}MB delta=${deltaMB.toFixed(1)}MB ` +
      `(500 load+release on 720p; known M4 Metal quirk — run with --expose-gc for eager reclaim)`,
  );
}

console.log(`image-cached: OK (${totalCases} correctness cases)`);
