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

console.log('image: OK');
