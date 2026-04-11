// image/image_cached.js — Phase 3b.1 persistent device buffer benchmark
//
// Four-path comparison for grayscale at 720p / 1080p / 4K:
//   1. JS loop              — baseline
//   2. Mojo CPU SIMD        — existing image.node grayscale
//   3. Mojo GPU one-shot    — existing image.node grayscaleGpu (alloc+H2D+kernel+D2H per call)
//   4. Mojo GPU cached      — new image_cached.node, loadImageGpu once, grayscaleHandle per call
//
// Unlike countByte (whose D2H is a scalar), grayscale still pays full D2H
// per call — the cached path only amortizes the one-time H2D upload and
// buffer allocation. Expected cached-vs-one-shot gap is smaller than for
// reductions. Break-even iteration count (vs one-shot) is printed to make
// the tradeoff legible.
//
// Build:  pixi run bash image/build.sh && pixi run bash image/build_cached.sh
// Run:    node image/image_cached.js

const addon = require('./build/image.node');

let cached = null;
try {
  cached = require('./build/image_cached.node');
} catch (e) {
  console.error('image_cached.node not found — run: pixi run bash image/build_cached.sh');
  process.exit(1);
}

// --- JS baseline -------------------------------------------------------------

function jsGrayscale(rgba, w, h) {
  const out = new Uint8Array(rgba.length);
  for (let i = 0; i < w * h; i++) {
    const off = i * 4;
    const gray = (77 * rgba[off] + 150 * rgba[off + 1] + 29 * rgba[off + 2]) >> 8;
    out[off] = out[off + 1] = out[off + 2] = gray;
    out[off + 3] = rgba[off + 3];
  }
  return out;
}

function createTestImage(w, h) {
  const rgba = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      rgba[i]     = (x * 255 / (w - 1)) | 0;
      rgba[i + 1] = (y * 255 / (h - 1)) | 0;
      rgba[i + 2] = Math.min(255, Math.sqrt((x - w/2)**2 + (y - h/2)**2) * 255 / (w/2)) | 0;
      rgba[i + 3] = 255;
    }
  }
  return rgba;
}

// --- Bench primitives --------------------------------------------------------

function bench(name, fn, iters) {
  for (let i = 0; i < Math.min(iters, 3); i++) fn();
  const start = performance.now();
  for (let i = 0; i < iters; i++) fn();
  const ms = performance.now() - start;
  const msPerOp = ms / iters;
  return { name, ms, msPerOp };
}

function formatSize(bytes) {
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + 'MB';
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(0) + 'KB';
  return bytes + 'B';
}

// --- Correctness spot-check --------------------------------------------------

{
  const w = 64, h = 64;
  const img = createTestImage(w, h);
  const jsOut = jsGrayscale(img, w, h);
  const cpuOut = addon.grayscale(img, w, h);
  const gpuOut = addon.grayscaleGpu(img, w, h);
  const dst = new Uint8Array(w * h * 4);
  const handle = cached.loadImageGpu(img, w, h);
  cached.grayscaleHandle(handle, dst);
  cached.releaseImageGpu(handle);

  const bytesEq = (a, b) => {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  };

  console.log('=== Correctness ===\n');
  const ok =
    bytesEq(cpuOut, jsOut) &&
    bytesEq(gpuOut, jsOut) &&
    bytesEq(dst, jsOut);
  console.log(`  cpuSimd=${bytesEq(cpuOut, jsOut)} gpuOneshot=${bytesEq(gpuOut, jsOut)} gpuCached=${bytesEq(dst, jsOut)}`);
  if (!ok) {
    console.error('  FAIL: mismatch');
    process.exit(1);
  }
  console.log('  PASS\n');
}

// --- Main benchmark loop -----------------------------------------------------

const sizes = [
  [1280,  720, 200],   // 720p,  3.7 MB,  200 iters
  [1920, 1080, 100],   // 1080p, 8.3 MB,  100 iters
  [3840, 2160,  20],   // 4K,   33 MB,    20 iters
];

console.log('--- grayscale benchmark (4 paths) ---\n');
console.log('  cached row reports per-call cost *after* loadImageGpu.\n');

for (const [W, H, ITERS] of sizes) {
  const bytes = W * H * 4;
  console.log(`=== grayscale: ${W}x${H} (${formatSize(bytes)}, ${ITERS} iters) ===\n`);
  const img = createTestImage(W, H);

  // 1. JS baseline
  const jsResult = bench('JS loop  ', () => jsGrayscale(img, W, H), ITERS);
  console.log(
    `  ${jsResult.name}: ${jsResult.msPerOp.toFixed(2)}ms/op  (baseline)`,
  );

  // 2. Mojo CPU SIMD
  const cpuResult = bench('Mojo SIMD', () => addon.grayscale(img, W, H), ITERS);
  const cpuSpeed = (jsResult.msPerOp / cpuResult.msPerOp).toFixed(1);
  console.log(
    `  ${cpuResult.name}: ${cpuResult.msPerOp.toFixed(2)}ms/op  ${cpuSpeed}x`,
  );

  // 3. Mojo GPU one-shot
  let gpuResult = null;
  try {
    const cpuSpot = addon.grayscale(img, W, H);
    const gpuSpot = addon.grayscaleGpu(img, W, H);
    for (let i = 0; i < cpuSpot.length; i++) {
      if (cpuSpot[i] !== gpuSpot[i]) {
        throw new Error(`byte ${i} mismatch: cpu=${cpuSpot[i]} gpu=${gpuSpot[i]}`);
      }
    }
    gpuResult = bench('GPU shot ', () => addon.grayscaleGpu(img, W, H), ITERS);
    const gSpeed = (jsResult.msPerOp / gpuResult.msPerOp).toFixed(1);
    console.log(
      `  ${gpuResult.name}: ${gpuResult.msPerOp.toFixed(2)}ms/op  ${gSpeed}x`,
    );
  } catch (e) {
    console.log(`  GPU shot : skipped (${e.message})`);
  }

  // 4. Mojo GPU cached
  try {
    const loadStart = performance.now();
    const handle = cached.loadImageGpu(img, W, H);
    const loadMs = performance.now() - loadStart;

    const dst = new Uint8Array(bytes);
    // Spot-check correctness before benchmarking.
    cached.grayscaleHandle(handle, dst);
    const cpuSpot = addon.grayscale(img, W, H);
    for (let i = 0; i < cpuSpot.length; i++) {
      if (cpuSpot[i] !== dst[i]) {
        cached.releaseImageGpu(handle);
        throw new Error(`byte ${i} mismatch: cpu=${cpuSpot[i]} cached=${dst[i]}`);
      }
    }

    const cachedResult = bench(
      'GPU cache',
      () => cached.grayscaleHandle(handle, dst),
      ITERS,
    );
    cached.releaseImageGpu(handle);

    const cSpeed = (jsResult.msPerOp / cachedResult.msPerOp).toFixed(1);
    let suffix = `   (loadImageGpu: ${loadMs.toFixed(1)}ms one-time)`;
    if (gpuResult && gpuResult.msPerOp > cachedResult.msPerOp) {
      const perCallSavings = gpuResult.msPerOp - cachedResult.msPerOp;
      const breakEven = Math.ceil(loadMs / perCallSavings);
      suffix += `   break-even vs one-shot: ${breakEven} iters`;
    }
    console.log(
      `  ${cachedResult.name}: ${cachedResult.msPerOp.toFixed(2)}ms/op  ${cSpeed}x${suffix}`,
    );
  } catch (e) {
    console.log(`  GPU cache: skipped (${e.message})`);
  }

  console.log('');
}
