// simd-search/search_cached.js — Phase 3a.1 persistent device buffer benchmark
//
// Four-row comparison at each size:
//   1. JS loop                 — baseline
//   2. Mojo CPU SIMD           — existing search.node countByte
//   3. Mojo GPU one-shot       — existing search.node countByteGpu (alloc+H2D+kernel+D2H+free per call)
//   4. Mojo GPU cached         — new search_cached.node, loadGpu once, countByteHandle per call
//
// The cached row tests the Phase 3 hypothesis: that amortizing PCIe transfer
// across N calls flips the Phase 2c result where GPU lost to CPU SIMD. If
// cached GPU is ≥20× JS at 17MB (ideally beating CPU SIMD), the API is
// validated and we extend the pattern to stats and grayscale.
//
// The cached row reports the per-call cost *after* loadGpu. The one-time
// loadGpu cost is printed separately for transparency.
//
// Build:  pixi run bash simd-search/build.sh && pixi run bash simd-search/build_cached.sh
// Run:    node simd-search/search_cached.js

const addon = require('./build/search.node');

let cached = null;
try {
  cached = require('./build/search_cached.node');
} catch (e) {
  console.error('search_cached.node not found — run: pixi run bash simd-search/build_cached.sh');
  process.exit(1);
}

// --- JS baseline -------------------------------------------------------------

function jsCountByte(buf, byte) {
  let count = 0;
  for (let i = 0; i < buf.length; i++) {
    if (buf[i] === byte) count++;
  }
  return count;
}

// --- Bench primitives --------------------------------------------------------

function bench(name, fn, iters) {
  for (let i = 0; i < Math.min(iters, 50); i++) fn();
  const start = performance.now();
  for (let i = 0; i < iters; i++) fn();
  const ms = performance.now() - start;
  const opsPerSec = Math.round(iters / (ms / 1000));
  return { name, ms, opsPerSec };
}

function formatOps(ops) {
  if (ops >= 1e6) return (ops / 1e6).toFixed(1) + 'M';
  if (ops >= 1e3) return (ops / 1e3).toFixed(1) + 'K';
  return ops.toString();
}

function formatSize(bytes) {
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(0) + 'MB';
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(0) + 'KB';
  return bytes + 'B';
}

function makeBuffer(size) {
  // ~1% newlines, matching search.js for comparable speedups.
  const buf = Buffer.alloc(size);
  for (let i = 0; i < size; i++) {
    buf[i] = Math.random() < 0.01 ? 0x0A : Math.floor(Math.random() * 255) + 1;
  }
  return buf;
}

// --- Correctness spot-check --------------------------------------------------

{
  const buf = Buffer.from('hello world hello');
  const target = 'l'.charCodeAt(0);
  const expected = 5;
  const jsVal = jsCountByte(buf, target);
  const cpuVal = addon.countByte(buf, target);
  const gpuOneshot = addon.countByteGpu(buf, target);
  const h = cached.loadGpu(buf);
  const gpuCached = cached.countByteHandle(h, target);
  cached.releaseGpu(h);
  console.log('=== Correctness ===\n');
  console.log(`  js=${jsVal} cpuSimd=${cpuVal} gpuOneshot=${gpuOneshot} gpuCached=${gpuCached} (expected ${expected})`);
  if ([jsVal, cpuVal, gpuOneshot, gpuCached].some((v) => v !== expected)) {
    console.error('  FAIL: mismatch');
    process.exit(1);
  }
  console.log('  PASS\n');
}

// --- Main benchmark loop -----------------------------------------------------

const sizes = [
  [1024, 100000],
  [65536, 10000],
  [1048576, 1000],
  [16777216, 200],
  [104857600, 20],
];

console.log('--- countByte benchmark (4 paths) ---\n');
console.log('  cached row reports per-call cost *after* loadGpu.\n');

for (const [SIZE, ITERS] of sizes) {
  console.log(`=== countByte: ${formatSize(SIZE)} buffer, ${ITERS} iters ===\n`);
  const buf = makeBuffer(SIZE);
  const target = 0x0A;

  // 1. JS baseline
  const jsResult = bench('JS loop  ', () => jsCountByte(buf, target), ITERS);
  console.log(
    `  ${jsResult.name}: ${jsResult.ms.toFixed(1)}ms  ${formatOps(jsResult.opsPerSec)} ops/sec  (baseline)`,
  );

  // 2. Mojo CPU SIMD
  const cpuResult = bench('Mojo SIMD', () => addon.countByte(buf, target), ITERS);
  const cpuSpeed = (jsResult.ms / cpuResult.ms).toFixed(1);
  console.log(
    `  ${cpuResult.name}: ${cpuResult.ms.toFixed(1)}ms  ${formatOps(cpuResult.opsPerSec)} ops/sec  ${cpuSpeed}x`,
  );

  // 3. Mojo GPU one-shot (existing addon)
  try {
    const cpuSpot = addon.countByte(buf, target);
    const gpuSpot = addon.countByteGpu(buf, target);
    if (cpuSpot !== gpuSpot) throw new Error(`mismatch: cpu=${cpuSpot} gpu=${gpuSpot}`);
    const gpuResult = bench('GPU shot ', () => addon.countByteGpu(buf, target), ITERS);
    const gSpeed = (jsResult.ms / gpuResult.ms).toFixed(1);
    console.log(
      `  ${gpuResult.name}: ${gpuResult.ms.toFixed(1)}ms  ${formatOps(gpuResult.opsPerSec)} ops/sec  ${gSpeed}x`,
    );
  } catch (e) {
    console.log(`  GPU shot : skipped (${e.message})`);
  }

  // 4. Mojo GPU cached (new prototype)
  try {
    // Time the one-time upload separately — it's not part of the per-call cost.
    const loadStart = performance.now();
    const h = cached.loadGpu(buf);
    const loadMs = performance.now() - loadStart;

    // Correctness spot-check before benchmarking.
    const cachedSpot = cached.countByteHandle(h, target);
    const cpuSpot = addon.countByte(buf, target);
    if (cachedSpot !== cpuSpot) {
      cached.releaseGpu(h);
      throw new Error(`mismatch: cached=${cachedSpot} cpu=${cpuSpot}`);
    }

    const cachedResult = bench(
      'GPU cache',
      () => cached.countByteHandle(h, target),
      ITERS,
    );
    const cSpeed = (jsResult.ms / cachedResult.ms).toFixed(1);
    cached.releaseGpu(h);

    console.log(
      `  ${cachedResult.name}: ${cachedResult.ms.toFixed(1)}ms  ${formatOps(cachedResult.opsPerSec)} ops/sec  ${cSpeed}x` +
        `   (loadGpu: ${loadMs.toFixed(1)}ms one-time)`,
    );
  } catch (e) {
    console.log(`  GPU cache: skipped (${e.message})`);
  }

  console.log('');
}
