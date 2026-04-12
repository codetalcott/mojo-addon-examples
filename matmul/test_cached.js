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

// Tolerance accommodates both FP32 (Metal, ~1e-7 per multiply) and TF32
// (H100 tensor cores, ~1e-3 per multiply, worst case K*1e-3 after summing).
// Important: JavaScript's jsMatmul accumulates in FP64 (Number is always
// double), so we're comparing FP64-accumulated reference to TF32 GPU
// result — TF32 will diverge at its precision floor regardless of what
// tolerance we pick per-element. GPU scheduling also makes the worst-case
// element non-deterministic. We handle this with rtol=1e-1 (typical TF32
// worst case for K<=256) PLUS a 1% outlier allowance (see assertMatchesWithOutliers
// below). For FP32-strict results, use the CPU matmul() path in matmul.node.
function closeEnough(a, b, rtol = 1e-1, atol = 1e-3) {
  const diff = Math.abs(a - b);
  return diff <= Math.max(rtol * Math.max(Math.abs(a), Math.abs(b)), atol);
}

// Allow up to `maxOutlierFraction` of elements to fail the per-element
// tolerance. This catches systematic bugs (many elements wrong) while
// tolerating the occasional TF32 outlier.
function assertMatchesWithOutliers(actual, expected, label, maxOutlierFraction = 0.01) {
  let failures = 0;
  let worstRel = 0;
  let worstIdx = -1;
  for (let i = 0; i < actual.length; i++) {
    if (!closeEnough(actual[i], expected[i])) {
      failures++;
      const denom = Math.max(Math.abs(actual[i]), Math.abs(expected[i]), 1e-12);
      const rel = Math.abs(actual[i] - expected[i]) / denom;
      if (rel > worstRel) { worstRel = rel; worstIdx = i; }
    }
  }
  const fraction = failures / actual.length;
  assert.ok(
    fraction <= maxOutlierFraction,
    `${label}: ${failures}/${actual.length} elements failed (${(fraction*100).toFixed(2)}%, ` +
      `worst rel err ${(worstRel*100).toFixed(2)}% at elem ${worstIdx}; ` +
      `max allowed ${(maxOutlierFraction*100).toFixed(2)}%)`,
  );
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

    assertMatchesWithOutliers(
      dst,
      expected,
      `${M}x${K}x${N} seed=0x${seed.toString(16)}`,
    );

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
  assertMatchesWithOutliers(dst, expected, `non-square ${M}x${K}x${N}`);
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
