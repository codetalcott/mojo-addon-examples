// matmul/matmul_rag.js — Phase 3d: RAG-shape cached matmul benchmark
//
// Measures cached GPU matmul on tall-skinny shapes that match local embedding
// retrieval workloads (query × corpus.T): [B, d] × [d, N] where
//   B = batch size (1 for single query, 64/256 for batched/offline)
//   d = embedding dim (768 for MiniLM/BGE-base, 1536 for OpenAI ada-002)
//   N = corpus size
//
// The headline Phase 3c number (28343× at 2048² square H100) doesn't translate
// directly to this shape — tall-skinny GEMM has lower arithmetic intensity and
// the D2H of the [B, N] scores matrix adds per-call cost. This bench produces
// the honest numbers for the RAG path.
//
// Flags:
//   --full         also run 1M-corpus shapes (multi-GB device memory)
//   --concurrency=N  run N concurrent matmulHandle calls (sync throughput test)
//
// Build:  pixi run bash matmul/build_cached.sh
// Run:    node matmul/matmul_rag.js

const path = require('path');

const args = process.argv.slice(2);
const FULL = args.includes('--full');
const SKIP_ORT = args.includes('--no-ort');
const concurrencyArg = args.find((a) => a.startsWith('--concurrency='));
const CONCURRENCY = concurrencyArg ? parseInt(concurrencyArg.split('=')[1], 10) : 0;

let cached = null;
try {
  cached = require('./build/matmul_cached.node');
} catch (e) {
  console.error('matmul_cached.node not found — run: pixi run bash matmul/build_cached.sh');
  process.exit(1);
}

// onnxruntime-node: optional CPU matmul baseline (apples-to-apples backend
// comparison). Skipped if --no-ort or if the package/model file is missing.
let ort = null;
let ortSession = null;
if (!SKIP_ORT) {
  try {
    ort = require('onnxruntime-node');
  } catch (e) {
    console.log('[info] onnxruntime-node not installed — run: npm install onnxruntime-node');
  }
}

// --- JS baseline — used only where flops budget allows ----------------------

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

function bench(fn, iters) {
  for (let i = 0; i < Math.min(iters, 2); i++) fn();
  const start = performance.now();
  for (let i = 0; i < iters; i++) fn();
  const ms = performance.now() - start;
  return { ms, msPerOp: ms / iters };
}

async function benchAsync(fn, iters) {
  for (let i = 0; i < Math.min(iters, 2); i++) await fn();
  const start = performance.now();
  for (let i = 0; i < iters; i++) await fn();
  const ms = performance.now() - start;
  return { ms, msPerOp: ms / iters };
}

function gflops(flops, msPerOp) {
  return (flops / (msPerOp / 1000)) / 1e9;
}

// --- Shapes ------------------------------------------------------------------

const BASE_SHAPES = [
  { B: 1,   d: 768,  N: 100_000, iters: 100, label: 'query × 100K corpus (d=768)' },
  { B: 1,   d: 1536, N: 100_000, iters: 50,  label: 'query × 100K corpus (d=1536)' },
  { B: 64,  d: 768,  N: 100_000, iters: 20,  label: 'batch-64 × 100K corpus' },
  { B: 256, d: 768,  N: 100_000, iters: 10,  label: 'batch-256 × 100K corpus (offline)' },
];

const FULL_SHAPES = [
  { B: 1,   d: 768,  N: 1_000_000, iters: 20, label: 'query × 1M corpus (d=768)' },
  { B: 1,   d: 1536, N: 1_000_000, iters: 10, label: 'query × 1M corpus (d=1536)' },
  { B: 64,  d: 768,  N: 1_000_000, iters: 5,  label: 'batch-64 × 1M corpus' },
];

const SHAPES = FULL ? [...BASE_SHAPES, ...FULL_SHAPES] : BASE_SHAPES;

// JS baseline cost budget: skip if > ~2 seconds estimated.
// Single-threaded Float32 triple-loop on modern CPU ≈ 150M flops/sec.
const JS_FLOPS_PER_SEC = 1.5e8;
const JS_BUDGET_SEC = 2.0;

// --- Correctness spot-check (small shape only) ------------------------------

function correctnessCheck() {
  const B = 4, d = 64, N = 1000;
  const a = makeMatrix(B, d);
  const b = makeMatrix(d, N);
  const jsC = jsMatmul(a, b, B, d, N);
  const hA = cached.loadMatrixGpu(a, B, d);
  const hB = cached.loadMatrixGpu(b, d, N);
  const dst = new Float32Array(B * N);
  cached.matmulHandle(hA, hB, dst);
  cached.releaseMatrixGpu(hA);
  cached.releaseMatrixGpu(hB);

  let mismatches = 0;
  for (let i = 0; i < dst.length; i++) {
    const diff = Math.abs(dst[i] - jsC[i]);
    const tol = Math.max(1e-1 * Math.max(Math.abs(dst[i]), Math.abs(jsC[i])), 1e-3);
    if (diff > tol) mismatches++;
  }
  const frac = mismatches / dst.length;
  console.log('=== Correctness (rectangular 4×64 × 64×1000) ===');
  console.log(`  mismatches: ${mismatches}/${dst.length} (${(frac * 100).toFixed(3)}%, rtol=1e-1)`);
  if (frac > 0.01) {
    console.error('  FAIL');
    process.exit(1);
  }
  console.log('  PASS\n');
}

// --- Main benchmark loop -----------------------------------------------------

async function benchShape({ B, d, N, iters, label }) {
  const flops = 2 * B * d * N;
  const bytesB = d * N * 4;
  const bytesDst = B * N * 4;

  console.log(`=== ${label} ===`);
  console.log(`  shape: [${B}, ${d}] × [${d}, ${N}]`);
  console.log(`  flops: ${(flops / 1e9).toFixed(2)} GFLOP/call`);
  console.log(`  device B: ${(bytesB / 1e6).toFixed(0)}MB, per-call dst: ${(bytesDst / 1e6).toFixed(1)}MB`);

  const a = makeMatrix(B, d);
  const b = makeMatrix(d, N);

  let jsResult = null;
  const jsEstSec = flops / JS_FLOPS_PER_SEC;
  if (jsEstSec <= JS_BUDGET_SEC) {
    jsResult = bench(() => jsMatmul(a, b, B, d, N), Math.max(1, Math.min(10, iters)));
    console.log(`  JS loop:      ${jsResult.msPerOp.toFixed(2)}ms/op  ${gflops(flops, jsResult.msPerOp).toFixed(2)} GFLOP/s  (baseline)`);
  } else {
    console.log(`  JS loop:      skipped (est ${jsEstSec.toFixed(1)}s > ${JS_BUDGET_SEC}s budget)`);
  }

  // onnxruntime-node CPU matmul (same computation, different backend).
  // Lazy: first shape initializes the session.
  let ortResult = null;
  if (ortSession) {
    const feeds = {
      A: new ort.Tensor('float32', a, [B, d]),
      B: new ort.Tensor('float32', b, [d, N]),
    };
    try {
      // Cap iterations for expensive shapes — ORT CPU on 256×768×100k is
      // seconds per call, not milliseconds.
      const ortIters = Math.min(iters, Math.max(1, Math.floor(500 / (flops / 1e9))));
      ortResult = await benchAsync(async () => await ortSession.run(feeds), ortIters);
      console.log(`  ORT CPU:      ${ortResult.msPerOp.toFixed(2)}ms/op  ${gflops(flops, ortResult.msPerOp).toFixed(2)} GFLOP/s  (onnxruntime-node MatMul)`);
    } catch (e) {
      console.log(`  ORT CPU:      failed (${e.message})`);
    }
  }

  try {
    const loadStart = performance.now();
    const hA = cached.loadMatrixGpu(a, B, d);
    const hB = cached.loadMatrixGpu(b, d, N);
    const loadMs = performance.now() - loadStart;

    const dst = new Float32Array(B * N);
    const cachedR = bench(() => cached.matmulHandle(hA, hB, dst), iters);
    const gpuGf = gflops(flops, cachedR.msPerOp);

    const parts = [];
    if (jsResult) parts.push(`${(jsResult.msPerOp / cachedR.msPerOp).toFixed(0)}× vs JS`);
    if (ortResult) parts.push(`${(ortResult.msPerOp / cachedR.msPerOp).toFixed(1)}× vs ORT`);
    const speedupNote = parts.length ? `  ${parts.join(', ')}` : '';
    console.log(`  GPU cached:   ${cachedR.msPerOp.toFixed(2)}ms/op  ${gpuGf.toFixed(2)} GFLOP/s${speedupNote}`);
    console.log(`                (loadMatrixGpu: ${loadMs.toFixed(1)}ms one-time)`);

    if (CONCURRENCY > 0) {
      // matmulHandle is synchronous, so "concurrency" here measures latency
      // distribution under a sustained burst of CONCURRENCY back-to-back
      // calls. True parallel dispatch from multiple threads would need the
      // async N-API path, which isn't wired up yet (see plan Step 4).
      const lat = new Float64Array(CONCURRENCY);
      // Warm.
      for (let c = 0; c < 3; c++) cached.matmulHandle(hA, hB, dst);
      const tStart = performance.now();
      for (let c = 0; c < CONCURRENCY; c++) {
        const t0 = performance.now();
        cached.matmulHandle(hA, hB, dst);
        lat[c] = performance.now() - t0;
      }
      const totalMs = performance.now() - tStart;
      const sorted = Float64Array.from(lat).sort();
      const pct = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
      const p50 = pct(0.50), p95 = pct(0.95), p99 = pct(0.99);
      const qps = (CONCURRENCY * 1000) / totalMs;
      console.log(`  burst=${CONCURRENCY}: p50=${p50.toFixed(2)}ms p95=${p95.toFixed(2)}ms p99=${p99.toFixed(2)}ms  ${qps.toFixed(0)} QPS  (ratio p99/p50=${(p99/p50).toFixed(1)}×)`);
    }

    cached.releaseMatrixGpu(hA);
    cached.releaseMatrixGpu(hB);
  } catch (e) {
    console.log(`  GPU cached:   skipped (${e.message})`);
  }
  console.log('');
}

// --- Entry -------------------------------------------------------------------

async function main() {
  console.log('--- matmul RAG-shape benchmark ---\n');
  console.log(`  mode: ${FULL ? 'full (includes 1M-corpus shapes)' : 'base (100K-corpus only; use --full for 1M)'}`);
  if (CONCURRENCY > 0) console.log(`  concurrency: ${CONCURRENCY} sync calls per batch`);

  if (ort) {
    try {
      const modelPath = path.resolve(__dirname, 'fixtures/matmul_dyn.onnx');
      ortSession = await ort.InferenceSession.create(modelPath);
      console.log(`  ORT baseline: onnxruntime-node ${ort.env?.versions?.common ?? ''} (CPU)`);
    } catch (e) {
      console.log(`  ORT baseline: unavailable (${e.message})`);
    }
  }
  console.log('');

  correctnessCheck();
  for (const shape of SHAPES) await benchShape(shape);
}

main().catch((e) => { console.error(e); process.exit(1); });
