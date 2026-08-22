const assert = require('node:assert');
const addon = require('./build/image.node');

// Single RGBA pixel: R=100, G=150, B=200, A=255
const pixel = new Uint8Array([100, 150, 200, 255]);
const gray = addon.grayscale(pixel, 1, 1);

// Integer grayscale: (77*100 + 150*150 + 29*200) >> 8 = 140
assert.strictEqual(gray[0], 140, 'grayscale R');
assert.strictEqual(gray[1], 140, 'grayscale G');
assert.strictEqual(gray[2], 140, 'grayscale B');
assert.strictEqual(gray[3], 255, 'grayscale A preserved');

const bright = addon.brightness(pixel, 1, 1, 0.5);
assert(bright[0] < pixel[0], 'brightness reduces');

// Brightness clamping: bright pixel * 2.0 should not exceed 255
const hotPixel = new Uint8Array([200, 200, 200, 255]);
const clamped = addon.brightness(hotPixel, 1, 1, 2.0);
assert(clamped[0] <= 255, 'brightness clamps R');
assert(clamped[1] <= 255, 'brightness clamps G');
assert(clamped[2] <= 255, 'brightness clamps B');
assert.strictEqual(clamped[3], 255, 'brightness preserves A');

// Threshold boundaries
const mid = new Uint8Array([128, 128, 128, 255]);
const thresh0 = addon.threshold(mid, 1, 1, 0);
assert.strictEqual(thresh0[0], 255, 'threshold 0 passes everything');
const thresh255 = addon.threshold(mid, 1, 1, 255);
assert.strictEqual(thresh255[0], 0, 'threshold 255 blocks everything');

// Blur: uniform image should remain unchanged
const white4x4 = new Uint8Array(4 * 4 * 4).fill(255);
const blurred = addon.blur(white4x4, 4, 4, 3);
assert.strictEqual(blurred[0], 255, 'blur uniform stays uniform');
assert.strictEqual(blurred[blurred.length - 1], 255, 'blur uniform last pixel');


// Parallel path: each kernel splits rows (or columns, for the blur's vertical
// pass) across workers. A 1x1 image gives every worker an empty range, so the
// row-splitting arithmetic is only really exercised at a realistic size.

// Deterministic PRNG so failures reproduce exactly.
function lcg(seed) {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return s / 4294967296; };
}

{
  const rnd = lcg(5);
  const W = 160, H = 120, n = W * H * 4;
  const rgba = new Uint8Array(n);
  for (let i = 0; i < n; i++) rgba[i] = Math.floor(rnd() * 256);
  for (let i = 3; i < n; i += 4) rgba[i] = 255;

  const g = addon.grayscale(rgba, W, H);
  const t = addon.threshold(rgba, W, H, 128);
  const b = addon.brightness(rgba, W, H, 1.5);
  const factorFp = Math.floor(1.5 * 256);
  for (let p = 0; p < W * H; p++) {
    const o = p * 4;
    const expGray = (77 * rgba[o] + 150 * rgba[o + 1] + 29 * rgba[o + 2]) >> 8;
    assert.strictEqual(g[o], expGray, `grayscale px ${p}`);
    assert.strictEqual(g[o + 3], rgba[o + 3], `grayscale alpha px ${p}`);
    assert.strictEqual(t[o], expGray >= 128 ? 255 : 0, `threshold px ${p}`);
    for (let c = 0; c < 3; c++) {
      assert.strictEqual(b[o + c], Math.min(255, (rgba[o + c] * factorFp) >> 8), `brightness px ${p} ch ${c}`);
    }
  }

  // Blur over both passes: a uniform image must survive unchanged.
  const uniform = new Uint8Array(n).fill(255);
  const bl = addon.blur(uniform, W, H, 5);
  for (let i = 0; i < n; i++) assert.strictEqual(bl[i], 255, `blur uniform at ${i}`);
  assert.strictEqual(addon.blur(rgba, W, H, 3).length, n, 'blur output length');
}

console.log('image: OK');
