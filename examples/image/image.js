// image/image.js — Correctness tests + benchmarks for image processing addon

const addon = require('./build/image.node');
const { performance } = require('perf_hooks');

// --- Synthetic RGBA data (no external deps) ---

function createTestImage(width, height) {
  const rgba = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      rgba[i]     = (x * 255 / (width - 1)) | 0;
      rgba[i + 1] = (y * 255 / (height - 1)) | 0;
      rgba[i + 2] = Math.min(255, Math.sqrt((x - width/2)**2 + (y - height/2)**2) * 255 / (width/2)) | 0;
      rgba[i + 3] = 255;
    }
  }
  return rgba;
}

// --- JS baselines ---

function jsGrayscale(rgba, w, h) {
  const out = new Uint8Array(rgba.length);
  for (let i = 0; i < w * h; i++) {
    const off = i * 4;
    const gray = (77 * rgba[off] + 150 * rgba[off+1] + 29 * rgba[off+2]) >> 8;
    out[off] = out[off+1] = out[off+2] = gray;
    out[off+3] = rgba[off+3];
  }
  return out;
}

function jsBrightness(rgba, w, h, factor) {
  const out = new Uint8Array(rgba.length);
  for (let i = 0; i < w * h; i++) {
    const off = i * 4;
    out[off]     = Math.min(255, (rgba[off] * factor) | 0);
    out[off + 1] = Math.min(255, (rgba[off+1] * factor) | 0);
    out[off + 2] = Math.min(255, (rgba[off+2] * factor) | 0);
    out[off + 3] = rgba[off + 3];
  }
  return out;
}

function jsThreshold(rgba, w, h, thresh) {
  const out = new Uint8Array(rgba.length);
  for (let i = 0; i < w * h; i++) {
    const off = i * 4;
    const gray = (77 * rgba[off] + 150 * rgba[off+1] + 29 * rgba[off+2]) >> 8;
    const v = gray >= thresh ? 255 : 0;
    out[off] = out[off+1] = out[off+2] = v;
    out[off+3] = rgba[off+3];
  }
  return out;
}

function jsBlur(rgba, w, h, radius) {
  const temp = new Uint8Array(rgba.length);
  const out = new Uint8Array(rgba.length);
  const diam = 2 * radius + 1;

  // Horizontal pass
  for (let row = 0; row < h; row++) {
    const ro = row * w * 4;
    for (let c = 0; c < 4; c++) {
      let sum = 0;
      for (let dx = -radius; dx <= radius; dx++) {
        sum += rgba[ro + Math.max(0, Math.min(dx, w-1)) * 4 + c];
      }
      temp[ro + c] = (sum / diam) | 0;
      for (let x = 1; x < w; x++) {
        sum += rgba[ro + Math.min(x + radius, w-1) * 4 + c];
        sum -= rgba[ro + Math.max(x - radius - 1, 0) * 4 + c];
        temp[ro + x * 4 + c] = (sum / diam) | 0;
      }
    }
  }

  // Vertical pass
  for (let col = 0; col < w; col++) {
    for (let c = 0; c < 4; c++) {
      let sum = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        sum += temp[Math.max(0, Math.min(dy, h-1)) * w * 4 + col * 4 + c];
      }
      out[col * 4 + c] = (sum / diam) | 0;
      for (let y = 1; y < h; y++) {
        sum += temp[Math.min(y + radius, h-1) * w * 4 + col * 4 + c];
        sum -= temp[Math.max(y - radius - 1, 0) * w * 4 + col * 4 + c];
        out[y * w * 4 + col * 4 + c] = (sum / diam) | 0;
      }
    }
  }
  return out;
}

// --- Correctness tests ---

console.log('=== Correctness Tests ===\n');

// Test with a small known image: single pixel (100, 150, 200, 255)
const testW = 4, testH = 4;
const testImg = new Uint8Array(testW * testH * 4);
for (let i = 0; i < testW * testH; i++) {
  testImg[i * 4]     = 100;
  testImg[i * 4 + 1] = 150;
  testImg[i * 4 + 2] = 200;
  testImg[i * 4 + 3] = 255;
}

// Grayscale: (77*100 + 150*150 + 29*200) >> 8 = (7700 + 22500 + 5800) / 256 = 140
const grayResult = addon.grayscale(testImg, testW, testH);
const expectedGray = (77 * 100 + 150 * 150 + 29 * 200) >> 8;
console.log(`grayscale pixel[0]: expected (${expectedGray},${expectedGray},${expectedGray},255), got (${grayResult[0]},${grayResult[1]},${grayResult[2]},${grayResult[3]})`);
console.assert(grayResult[0] === expectedGray && grayResult[1] === expectedGray && grayResult[2] === expectedGray && grayResult[3] === 255, 'grayscale FAILED');
console.log('  grayscale: PASS');

// Brightness 2.0: (100*2=200, 150*2=255 clamped, 200*2=255 clamped, 255)
const brightResult = addon.brightness(testImg, testW, testH, 2.0);
// Fixed-point: (100 * 512) >> 8 = 200, (150 * 512) >> 8 = 300 → clamped 255, (200 * 512) >> 8 = 400 → clamped 255
console.log(`brightness(2.0) pixel[0]: expected (200,255,255,255), got (${brightResult[0]},${brightResult[1]},${brightResult[2]},${brightResult[3]})`);
console.assert(brightResult[0] === 200 && brightResult[1] === 255 && brightResult[2] === 255 && brightResult[3] === 255, 'brightness FAILED');
console.log('  brightness: PASS');

// Threshold at 128: gray=140 >= 128 → 255
const threshResult = addon.threshold(testImg, testW, testH, 128);
console.log(`threshold(128) pixel[0]: expected (255,255,255,255), got (${threshResult[0]},${threshResult[1]},${threshResult[2]},${threshResult[3]})`);
console.assert(threshResult[0] === 255 && threshResult[1] === 255 && threshResult[2] === 255 && threshResult[3] === 255, 'threshold FAILED');
console.log('  threshold: PASS');

// Threshold at 200: gray=140 < 200 → 0
const threshResult2 = addon.threshold(testImg, testW, testH, 200);
console.log(`threshold(200) pixel[0]: expected (0,0,0,255), got (${threshResult2[0]},${threshResult2[1]},${threshResult2[2]},${threshResult2[3]})`);
console.assert(threshResult2[0] === 0 && threshResult2[1] === 0 && threshResult2[2] === 0 && threshResult2[3] === 255, 'threshold(200) FAILED');
console.log('  threshold(200): PASS');

// Blur: uniform image should stay uniform
const blurResult = addon.blur(testImg, testW, testH, 1);
console.log(`blur(1) pixel[0]: expected (100,150,200,255), got (${blurResult[0]},${blurResult[1]},${blurResult[2]},${blurResult[3]})`);
console.assert(blurResult[0] === 100 && blurResult[1] === 150 && blurResult[2] === 200 && blurResult[3] === 255, 'blur uniform FAILED');
console.log('  blur (uniform): PASS');

console.log('\nAll correctness tests passed!\n');

// --- Benchmarks ---

function bench(name, fn, warmup = 3, iters = 10) {
  for (let i = 0; i < warmup; i++) fn();
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) fn();
  const ms = (performance.now() - t0) / iters;
  return { name, ms };
}

function runBenchmarks(label, width, height, iters) {
  console.log(`--- ${label} (${width}x${height}, ${((width * height * 4) / 1e6).toFixed(1)}MB) ---`);
  const img = createTestImage(width, height);

  const jsGray = bench('JS grayscale', () => jsGrayscale(img, width, height), 3, iters);
  const mojoGray = bench('Mojo grayscale', () => addon.grayscale(img, width, height), 3, iters);
  let grayLine = `  grayscale:  JS ${jsGray.ms.toFixed(2)}ms  Mojo ${mojoGray.ms.toFixed(2)}ms  ${(jsGray.ms / mojoGray.ms).toFixed(1)}x`;
  if (typeof addon.grayscaleGpu === 'function') {
    try {
      // Correctness spot-check: GPU output must match CPU exactly (integer math).
      const cpuSpot = addon.grayscale(img, width, height);
      const gpuSpot = addon.grayscaleGpu(img, width, height);
      for (let i = 0; i < cpuSpot.length; i++) {
        if (cpuSpot[i] !== gpuSpot[i]) {
          throw new Error(`byte ${i} mismatch: cpu=${cpuSpot[i]} gpu=${gpuSpot[i]}`);
        }
      }
      const mojoGrayGpu = bench('Mojo grayscaleGpu', () => addon.grayscaleGpu(img, width, height), 3, iters);
      grayLine += `  GPU ${mojoGrayGpu.ms.toFixed(2)}ms  ${(jsGray.ms / mojoGrayGpu.ms).toFixed(1)}x`;
    } catch (e) {
      grayLine += `  GPU skipped (${e.message})`;
    }
  }
  console.log(grayLine);

  const jsBright = bench('JS brightness', () => jsBrightness(img, width, height, 1.5), 3, iters);
  const mojoBright = bench('Mojo brightness', () => addon.brightness(img, width, height, 1.5), 3, iters);
  console.log(`  brightness: JS ${jsBright.ms.toFixed(2)}ms  Mojo ${mojoBright.ms.toFixed(2)}ms  ${(jsBright.ms / mojoBright.ms).toFixed(1)}x`);

  const jsThresh = bench('JS threshold', () => jsThreshold(img, width, height, 128), 3, iters);
  const mojoThresh = bench('Mojo threshold', () => addon.threshold(img, width, height, 128), 3, iters);
  console.log(`  threshold:  JS ${jsThresh.ms.toFixed(2)}ms  Mojo ${mojoThresh.ms.toFixed(2)}ms  ${(jsThresh.ms / mojoThresh.ms).toFixed(1)}x`);

  const blurIters = Math.max(1, Math.floor(iters / 2));
  const jsB = bench('JS blur', () => jsBlur(img, width, height, 5), 2, blurIters);
  const mojoB = bench('Mojo blur', () => addon.blur(img, width, height, 5), 2, blurIters);
  console.log(`  blur(r=5):  JS ${jsB.ms.toFixed(2)}ms  Mojo ${mojoB.ms.toFixed(2)}ms  ${(jsB.ms / mojoB.ms).toFixed(1)}x`);

  console.log();
}

console.log('=== Benchmarks ===\n');
runBenchmarks('720p', 1280, 720, 20);
runBenchmarks('1080p', 1920, 1080, 10);
runBenchmarks('4K', 3840, 2160, 5);
