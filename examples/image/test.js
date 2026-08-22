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

// --- Realistic-size parallel-path coverage ------------------------------------
// Every image kernel dispatches through a `parallelize_safe` worker split
// (there is no PARALLEL_THRESHOLD here — the parallel path is always taken).
// The tiny fixtures above cannot detect a wrong split: a 1x1 image gives every
// worker a degenerate range, and a uniform 4x4 white image returns 255 no matter
// which rows or columns a worker actually visited. These cases use non-uniform
// images large enough that a dropped, duplicated, or misaligned row/column strip
// changes the output, and check every byte against a JS reference.
//
// Sizes are chosen in pairs: one that divides evenly by NUM_WORKERS (4) and one
// that does not, so the `wid < NUM_WORKERS - 1 else <dim>` remainder branch —
// where the last worker absorbs the leftover rows/columns — is exercised too.

const SIZES = [
  { width: 160, height: 120 }, // 160/4 and 120/4 are exact
  { width: 157, height: 113 }, // remainder lands on the last worker
];

// Deterministic pseudo-random image — no seeded-RNG dependency, and every
// channel varies independently so a channel mix-up is visible.
function makeImage(width, height) {
  const px = new Uint8Array(width * height * 4);
  let seed = 0x2f6e2b1;
  for (let i = 0; i < px.length; i++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    px[i] = (seed >> 16) & 0xff;
  }
  return px;
}

function assertBytesEqual(actual, expected, label) {
  assert.strictEqual(actual.length, expected.length, `${label}: length`);
  for (let i = 0; i < expected.length; i++) {
    if (actual[i] !== expected[i]) {
      const px = Math.floor(i / 4);
      assert.fail(
        `${label}: byte ${i} (pixel ${px}, channel ${i % 4}) ` +
        `expected ${expected[i]}, got ${actual[i]}`
      );
    }
  }
}

function grayscaleRef(src) {
  const out = new Uint8Array(src.length);
  for (let i = 0; i < src.length; i += 4) {
    const g = (77 * src[i] + 150 * src[i + 1] + 29 * src[i + 2]) >> 8;
    out[i] = g; out[i + 1] = g; out[i + 2] = g; out[i + 3] = src[i + 3];
  }
  return out;
}

function brightnessRef(src, factor) {
  const fp = Math.trunc(factor * 256.0); // matches UInt32(factor * 256.0)
  const out = new Uint8Array(src.length);
  for (let i = 0; i < src.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      out[i + c] = Math.min(255, Math.floor((src[i + c] * fp) / 256));
    }
    out[i + 3] = src[i + 3];
  }
  return out;
}

function thresholdRef(src, thresh) {
  const out = new Uint8Array(src.length);
  for (let i = 0; i < src.length; i += 4) {
    const g = (77 * src[i] + 150 * src[i + 1] + 29 * src[i + 2]) >> 8;
    const v = g >= thresh ? 255 : 0;
    out[i] = v; out[i + 1] = v; out[i + 2] = v; out[i + 3] = src[i + 3];
  }
  return out;
}

// Separable box blur with clamped edges, all four channels (alpha included, as
// the kernel does). The kernel slides a running window; sliding and direct
// summation agree exactly under clamping, so a direct sum is a valid reference.
function blurRef(src, width, height, radius) {
  const diameter = 2 * radius + 1;
  const temp = new Uint8Array(src.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < 4; c++) {
        let sum = 0;
        for (let dx = -radius; dx <= radius; dx++) {
          let sx = x + dx;
          if (sx < 0) sx = 0;
          if (sx >= width) sx = width - 1;
          sum += src[(y * width + sx) * 4 + c];
        }
        temp[(y * width + x) * 4 + c] = Math.floor(sum / diameter);
      }
    }
  }
  const out = new Uint8Array(src.length);
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      for (let c = 0; c < 4; c++) {
        let sum = 0;
        for (let dy = -radius; dy <= radius; dy++) {
          let sy = y + dy;
          if (sy < 0) sy = 0;
          if (sy >= height) sy = height - 1;
          sum += temp[(sy * width + x) * 4 + c];
        }
        out[(y * width + x) * 4 + c] = Math.floor(sum / diameter);
      }
    }
  }
  return out;
}

for (const { width, height } of SIZES) {
  const dims = `${width}x${height}`;
  const img = makeImage(width, height);

  assertBytesEqual(
    addon.grayscale(img, width, height), grayscaleRef(img), `grayscale ${dims}`
  );
  assertBytesEqual(
    addon.brightness(img, width, height, 1.4), brightnessRef(img, 1.4),
    `brightness ${dims}`
  );
  assertBytesEqual(
    addon.threshold(img, width, height, 128), thresholdRef(img, 128),
    `threshold ${dims}`
  );

  // Blur is the one with two independent splits — rows for the horizontal pass,
  // columns for the vertical pass — so check both a small and a large radius.
  for (const radius of [1, 3]) {
    assertBytesEqual(
      addon.blur(img, width, height, radius), blurRef(img, width, height, radius),
      `blur ${dims} r=${radius}`
    );
  }

  // A uniform image must survive the blur untouched. This catches a worker
  // reading outside its strip even when the reference check somehow agrees:
  // every averaged window is 200, so any deviation means a stray read.
  const uniform = new Uint8Array(width * height * 4).fill(200);
  const uniformBlurred = addon.blur(uniform, width, height, 3);
  for (let i = 0; i < uniformBlurred.length; i++) {
    assert.strictEqual(
      uniformBlurred[i], 200, `blur ${dims} uniform: byte ${i} changed`
    );
  }
}

console.log('image: OK');
