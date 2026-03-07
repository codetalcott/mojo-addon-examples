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

console.log('image: OK');
