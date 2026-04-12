// matmul/matmul_cached.js — Phase 3c.2 persistent device buffer matmul benchmark
//
// Three-path comparison at each size:
//   1. JS loop                — triple-loop baseline (single-threaded)
//   2. Mojo CPU parallel      — existing matmul.node matmulParallel (vectorize + parallelize)
//   3. Mojo GPU cached        — new matmul_cached.node, loadMatrixGpu once, matmulHandle per call
//
// The cached path uses MAX's linalg.matmul which dispatches to tensor cores
// on H100 (TF32 for FP32 inputs) and Metal on Apple silicon. Per-call cost
// is kernel + D2H for the C matrix; A and B stay on-device.
//
// Build:  pixi run bash matmul/build.sh && pixi run bash matmul/build_cached.sh
// Run:    node matmul/matmul_cached.js

const addon = require('./build/matmul.node');

let cached = null;
try {
  cached = require('./build/matmul_cached.node');
} catch (e) {
  console.error('matmul_cached.node not found — run: pixi run bash matmul/build_cached.sh');
  process.exit(1);
}

// --- JS baseline (Float64 → matches existing matmul.js, but we use Float32
// here to compare like-for-like with the GPU FP32 path) ---------------------

function jsMatmul(a, b, M, K, N) {
  const c = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let p = 0; p < K; p++) {
        sum += a[i * K + p] * b[p * N + j];
      }
      c[i * N + j] = sum;
    }
  }
  return c;
}

function makeMatrix(rows, cols) {
  const m = new Float32Array(rows * cols);
  for (let i = 0; i < m.length; i++) m[i] = Math.random() * 2 - 1;
  return m;
}

// --- Bench primitives --------------------------------------------------------

function bench(name, fn, iters) {
  for (let i = 0; i < Math.min(iters, 2); i++) fn();
  const start = performance.now();
  for (let i = 0; i < iters; i++) fn();
  const ms = performance.now() - start;
  const msPerOp = ms / iters;
  return { name, ms, msPerOp };
}

// --- Correctness spot-check --------------------------------------------------

{
  const N = 64;
  const a = makeMatrix(N, N);
  const b = makeMatrix(N, N);
  const jsC = jsMatmul(a, b, N, N, N);
  const hA = cached.loadMatrixGpu(a, N, N);
  const hB = cached.loadMatrixGpu(b, N, N);
  const dst = new Float32Array(N * N);
  cached.matmulHandle(hA, hB, dst);
  cached.releaseMatrixGpu(hA);
  cached.releaseMatrixGpu(hB);

  let mismatches = 0;
  for (let i = 0; i < dst.length; i++) {
    const diff = Math.abs(dst[i] - jsC[i]);
    if (diff > Math.max(1e-3 * Math.max(Math.abs(dst[i]), Math.abs(jsC[i])), 1e-5)) {
      mismatches++;
    }
  }
  console.log('=== Correctness ===\n');
  console.log(`  64x64 mismatches: ${mismatches}/${dst.length} (rtol=1e-3, atol=1e-5)`);
  if (mismatches > 0) {
    console.error('  FAIL');
    process.exit(1);
  }
  console.log('  PASS\n');
}

// --- Main benchmark loop -----------------------------------------------------

// Note: JS baseline is Float32 triple-loop (single-threaded).
// Mojo CPU parallel is Float64 (existing matmul.node uses Float64Array).
// GPU cached is FP32.
// The CPU parallel column uses Float64 because the existing addon only
// exposes Float64 matmul. It's still a valid comparison — Float64 is
// slightly slower than Float32 on CPU, so the GPU-vs-CPU speedup ratio
// is conservative.

const sizes = [
  [256,   100],
  [512,    20],
  [1024,    5],
  [2048,    2],
];

console.log('--- matmul benchmark (3 paths) ---\n');
console.log('  cached row reports per-call cost *after* loadMatrixGpu.\n');

for (const [N, ITERS] of sizes) {
  const a32 = makeMatrix(N, N);
  const b32 = makeMatrix(N, N);
  console.log(`=== matmul: ${N}x${N} (${ITERS} iters) ===\n`);

  // 1. JS baseline (Float32)
  const jsR = bench('JS loop  ', () => jsMatmul(a32, b32, N, N, N), ITERS);
  console.log(`  ${jsR.name}: ${jsR.msPerOp.toFixed(2)}ms/op  (baseline)`);

  // 2. Mojo CPU parallel (Float64 — existing addon)
  const a64 = new Float64Array(a32);
  const b64 = new Float64Array(b32);
  const c64 = new Float64Array(N * N);
  const cpuR = bench('Mojo CPU ', () => addon.matmulParallel(a64, b64, c64, N, N, N), ITERS);
  const cpuSpeed = (jsR.msPerOp / cpuR.msPerOp).toFixed(1);
  console.log(`  ${cpuR.name}: ${cpuR.msPerOp.toFixed(2)}ms/op  ${cpuSpeed}x`);

  // 3. GPU cached (FP32)
  try {
    const loadStart = performance.now();
    const hA = cached.loadMatrixGpu(a32, N, N);
    const hB = cached.loadMatrixGpu(b32, N, N);
    const loadMs = performance.now() - loadStart;

    const dst = new Float32Array(N * N);
    // Spot-check correctness before benchmarking.
    cached.matmulHandle(hA, hB, dst);
    const jsC = jsMatmul(a32, b32, N, N, N);
    let mismatches = 0;
    for (let i = 0; i < dst.length; i++) {
      const diff = Math.abs(dst[i] - jsC[i]);
      if (diff > Math.max(1e-3 * Math.max(Math.abs(dst[i]), Math.abs(jsC[i])), 1e-4)) {
        mismatches++;
      }
    }
    if (mismatches > dst.length * 0.01) {
      cached.releaseMatrixGpu(hA);
      cached.releaseMatrixGpu(hB);
      throw new Error(`too many mismatches: ${mismatches}/${dst.length}`);
    }

    const cachedR = bench(
      'GPU cache',
      () => cached.matmulHandle(hA, hB, dst),
      ITERS,
    );
    cached.releaseMatrixGpu(hA);
    cached.releaseMatrixGpu(hB);

    const cSpeed = (jsR.msPerOp / cachedR.msPerOp).toFixed(1);
    let suffix = `   (loadMatrixGpu: ${loadMs.toFixed(1)}ms one-time)`;
    console.log(
      `  ${cachedR.name}: ${cachedR.msPerOp.toFixed(2)}ms/op  ${cSpeed}x${suffix}`,
    );
  } catch (e) {
    console.log(`  GPU cache: skipped (${e.message})`);
  }

  console.log('');
}
