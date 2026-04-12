// Regression test for the Phase 3c.2 cached matmul addon.
//
// Verifies loadMatrixGpu / matmulHandle / releaseMatrixGpu produce results
// that match a plain JS matmul within Float32 precision tolerance, across
// a range of sizes. Also runs stability (100 repeat queries) and tests
// the two-handle release + tombstone behavior.
//
// Run: node matmul/test_cached.js

const assert = require('node:assert');

let cached;
try {
  cached = require('./build/matmul_cached.node');
} catch (e) {
  console.error('matmul_cached.node not found — run: pixi run bash matmul/build_cached.sh');
  process.exit(1);
}

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

function makeMatrix(rows, cols, seed) {
  const m = new Float32Array(rows * cols);
  let s = seed | 0;
  for (let i = 0; i < m.length; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    m[i] = (s / 0x7fffffff) * 2 - 1; // [-1, 1]
  }
  return m;
}

function closeEnough(a, b, rtol = 1e-3, atol = 1e-5) {
  const diff = Math.abs(a - b);
  return diff <= Math.max(rtol * Math.max(Math.abs(a), Math.abs(b)), atol);
}

// --- Correctness: 4 sizes × 2 seeds = 8 cases ------------------------------

const sizes = [
  [16, 16, 16],
  [64, 64, 64],
  [128, 128, 128],
  [256, 256, 256],
];
const seeds = [0xC0FFEE, 0xBADF00D];

let totalCases = 0;
for (const [M, K, N] of sizes) {
  for (const seed of seeds) {
    const a = makeMatrix(M, K, seed);
    const b = makeMatrix(K, N, seed + 1);
    const expected = jsMatmul(a, b, M, K, N);

    const hA = cached.loadMatrixGpu(a, M, K);
    const hB = cached.loadMatrixGpu(b, K, N);
    const dst = new Float32Array(M * N);
    cached.matmulHandle(hA, hB, dst);

    // FP32 GPU matmul should match JS FP32 within 1e-3 relative tolerance.
    // (TF32 on H100 has ~1e-3 precision; Metal FP32 should be exact.)
    for (let i = 0; i < dst.length; i++) {
      assert.ok(
        closeEnough(dst[i], expected[i]),
        `${M}x${K}x${N} seed=0x${seed.toString(16)} elem ${i}: ` +
          `expected=${expected[i]} got=${dst[i]}`,
      );
    }

    cached.releaseMatrixGpu(hA);
    cached.releaseMatrixGpu(hB);

    // After release, matmulHandle should throw.
    let threw = false;
    try {
      cached.matmulHandle(hA, hB, dst);
    } catch (e) {
      threw = true;
    }
    assert.ok(threw, `${M}x${K}x${N}: matmulHandle after release should throw`);
    totalCases++;
  }
}

// --- Non-square test --------------------------------------------------------
{
  const M = 64, K = 128, N = 32;
  const a = makeMatrix(M, K, 0x5EED);
  const b = makeMatrix(K, N, 0x5EEE);
  const expected = jsMatmul(a, b, M, K, N);
  const hA = cached.loadMatrixGpu(a, M, K);
  const hB = cached.loadMatrixGpu(b, K, N);
  const dst = new Float32Array(M * N);
  cached.matmulHandle(hA, hB, dst);
  for (let i = 0; i < dst.length; i++) {
    assert.ok(
      closeEnough(dst[i], expected[i]),
      `non-square ${M}x${K}x${N} elem ${i}: expected=${expected[i]} got=${dst[i]}`,
    );
  }
  cached.releaseMatrixGpu(hA);
  cached.releaseMatrixGpu(hB);
  totalCases++;
}

// --- Dimension mismatch should throw ----------------------------------------
{
  const a = makeMatrix(32, 64, 0x1111);
  const b = makeMatrix(32, 32, 0x2222); // B.rows=32 != A.cols=64
  const hA = cached.loadMatrixGpu(a, 32, 64);
  const hB = cached.loadMatrixGpu(b, 32, 32);
  const dst = new Float32Array(32 * 32);
  let threw = false;
  try {
    cached.matmulHandle(hA, hB, dst);
  } catch (e) {
    threw = true;
  }
  assert.ok(threw, 'dimension mismatch should throw');
  cached.releaseMatrixGpu(hA);
  cached.releaseMatrixGpu(hB);
  totalCases++;
}

// --- Stability: 100 repeated matmuls on same handles -----------------------
{
  const M = 128, K = 128, N = 128;
  const a = makeMatrix(M, K, 0xBEEF);
  const b = makeMatrix(K, N, 0xCAFE);
  const hA = cached.loadMatrixGpu(a, M, K);
  const hB = cached.loadMatrixGpu(b, K, N);
  const dst = new Float32Array(M * N);
  let firstVal = null;
  for (let i = 0; i < 100; i++) {
    cached.matmulHandle(hA, hB, dst);
    if (i === 0) firstVal = dst[0];
    assert.strictEqual(dst[0], firstVal, `stability iter ${i}: C[0] drifted`);
  }
  cached.releaseMatrixGpu(hA);
  cached.releaseMatrixGpu(hB);
  totalCases += 100;
}

console.log(`matmul-cached: OK (${totalCases} correctness cases)`);
