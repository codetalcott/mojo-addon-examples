// matmul/matmul.js — Progressive matrix multiply benchmark
//
// Demonstrates Mojo's optimization story: naive -> SIMD -> tiled -> parallel
// Each step maps to one Mojo language feature with measurable speedup.
//
// Build:  pixi run bash matmul/build.sh
// Run:    node matmul/matmul.js

const addon = require('./build/matmul.node');

// --- Pure JS baseline (naive i,j,k) -----------------------------------------

function jsMatmul(a, b, out, M, K, N) {
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let p = 0; p < K; p++) {
        sum += a[i * K + p] * b[p * N + j];
      }
      out[i * N + j] = sum;
    }
  }
}

// --- Correctness check -------------------------------------------------------

console.log('=== Correctness ===\n');

const M = 4, K = 3, N = 2;
const a = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
const b = new Float64Array([1, 2, 3, 4, 5, 6]);
const expected = new Float64Array(M * N);
jsMatmul(a, b, expected, M, K, N);

const variants = ['matmulNaive', 'matmulVectorized', 'matmulTiled', 'matmulParallel'];

for (const name of variants) {
  const out = new Float64Array(M * N);
  addon[name](a, b, out, M, K, N);
  const match = expected.every((v, i) => Math.abs(v - out[i]) < 1e-10);
  console.log(`  ${name}: ${match ? 'PASS' : 'FAIL'}`);
  if (!match) {
    console.log('    expected:', Array.from(expected));
    console.log('    got:     ', Array.from(out));
  }
}

// --- Benchmark ---------------------------------------------------------------

function bench(name, fn, iters) {
  // warmup
  for (let i = 0; i < Math.min(iters, 10); i++) fn();

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

// Benchmark sizes: [N, iters]
const sizes = [
  [128, 500],
  [256, 100],
  [512, 20],
  [1024, 5],
  [2048, 1],
];

for (const [DIM, ITERS] of sizes) {
  console.log(`\n=== ${DIM}x${DIM} matmul (${(2 * DIM ** 3 / 1e6).toFixed(1)}M FLOPs) ===\n`);

  const va = new Float64Array(DIM * DIM);
  const vb = new Float64Array(DIM * DIM);
  const vout = new Float64Array(DIM * DIM);

  for (let i = 0; i < DIM * DIM; i++) {
    va[i] = Math.random() * 2 - 1;
    vb[i] = Math.random() * 2 - 1;
  }

  // JS baseline
  const jsResult = bench('JS naive     ', () => jsMatmul(va, vb, vout, DIM, DIM, DIM), ITERS);

  // Mojo variants
  const results = [jsResult];
  for (const name of variants) {
    const padded = (name.replace('matmul', 'Mojo ') + '       ').slice(0, 14);
    results.push(bench(padded, () => addon[name](va, vb, vout, DIM, DIM, DIM), ITERS));
  }

  // Print results table
  const jsMs = jsResult.ms;
  for (const r of results) {
    const speedup = (jsMs / r.ms).toFixed(1);
    const marker = r === jsResult ? '(baseline)' : `${speedup}x`;
    console.log(`  ${r.name}: ${r.ms.toFixed(1)}ms  ${formatOps(r.opsPerSec)} ops/sec  ${marker}`);
  }
}
