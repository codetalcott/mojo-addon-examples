// scripts/tf32-error-sweep.js — characterise matmulHandle's numeric error.
//
// packages/retrieve's tolerance floor is derived from a model:
//
//     ATOL = 4 * sqrt(K) * 2**-11
//
// TF32 (H100's path for FP32 inputs) keeps ~11 mantissa bits, and a K-term dot
// product accumulates rounding by random walk, so absolute error should grow
// like sqrt(K)*eps and NOT shrink with the result. That model was fitted to a
// single observation — max |abs err| 2.981e-3 at K=64, one seed. A single
// sample cannot distinguish real headroom from a lucky draw, and the max over
// 4000 elements is itself a random variable with a tail.
//
// This sweeps seeds (distribution) and K (the sqrt(K) claim), so the model is
// tested rather than the number re-observed.
//
// Usage:
//   pixi run node scripts/tf32-error-sweep.js [--seeds N] [--quick]
//
// Reads build/retrieve.node directly — no Jest — so it costs seconds once the
// addon is built. Prints a table plus a verdict on whether ATOL holds.

const path = require('path');

const ADDON = path.join(__dirname, '..', 'packages', 'retrieve', 'build', 'retrieve.node');
let gpu;
try {
  gpu = require(ADDON);
} catch (e) {
  console.error(`FATAL: cannot load ${ADDON}\n  ${e.message}`);
  console.error('  run: pixi run bash packages/retrieve/build.sh');
  process.exit(1);
}

const argv = process.argv.slice(2);
const arg = (name, dflt) => {
  const i = argv.indexOf(name);
  return i === -1 ? dflt : Number(argv[i + 1]);
};
const QUICK = argv.includes('--quick');
const SEEDS = arg('--seeds', QUICK ? 5 : 20);
const K_VALUES = QUICK ? [64] : [16, 32, 64, 128, 256];
const M = 4;
const N = 1000;

// Same generator as packages/retrieve/tests/retrieve.test.js, so numbers here
// are comparable with a failing test run.
function mulberry32(seed) {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// float64 accumulation, matching the test's reference. Deliberately more
// precise than the GPU so the measured gap is the GPU's error, not a race
// between two approximations.
function jsMatmul(a, b, M, K, N) {
  const c = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let p = 0; p < K; p++) sum += a[i * K + p] * b[p * N + j];
      c[i * N + j] = sum;
    }
  }
  return c;
}

const TF32_EPS = 2 ** -11;
const atolFor = (K) => 4 * Math.sqrt(K) * TF32_EPS;

// RTOL only governs where the relative term exceeds the floor, i.e. where
//     RTOL * |e| > ATOL   <=>   |e| > ATOL / RTOL
// Below that crossover ATOL decides the comparison, so relative error sampled
// there says nothing about how tight RTOL could be.
//
// The first version of this tool sampled every |e| > 1e-2 and reported max
// relative errors of 0.10–0.27 on H100, concluding "do NOT tighten RTOL below
// 2.7e+0" — 270%, which is nonsense. At K=256 the crossover is |e| > 0.313
// while the threshold was 1e-2, so the statistic was dominated by small
// elements that ATOL rescues. Measuring in the wrong regime produces a
// confident number about the wrong question.
const RTOL_CURRENT = 1e-1;
const relCrossover = (K) => atolFor(K) / RTOL_CURRENT;

function runOne(K, seed) {
  const r = mulberry32(seed);
  const rnd = () => r() * 2 - 1;
  const a = new Float32Array(M * K);
  const b = new Float32Array(K * N);
  for (let i = 0; i < a.length; i++) a[i] = rnd();
  for (let i = 0; i < b.length; i++) b[i] = rnd();

  const expected = jsMatmul(a, b, M, K, N);
  const dst = new Float32Array(M * N);
  const hA = gpu.loadMatrixGpu(a, M, K);
  const hB = gpu.loadMatrixGpu(b, K, N);
  gpu.matmulHandle(hA, hB, dst);
  gpu.releaseMatrixGpu(hA);
  gpu.releaseMatrixGpu(hB);

  let maxAbs = 0;
  let maxRel = 0;
  let relSamples = 0;
  const crossover = relCrossover(K);
  for (let i = 0; i < dst.length; i++) {
    const e = expected[i];
    const abs = Math.abs(dst[i] - e);
    if (abs > maxAbs) maxAbs = abs;
    // Only where RTOL actually governs — see relCrossover above.
    if (Math.abs(e) > crossover) {
      maxRel = Math.max(maxRel, abs / Math.abs(e));
      relSamples++;
    }
  }
  return { maxAbs, maxRel, relSamples };
}

const pct = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];

console.log(`tf32-error-sweep  M=${M} N=${N}  seeds=${SEEDS}  K=[${K_VALUES.join(', ')}]`);
console.log('');
console.log('   K   seeds   max|abs|    p50|abs|    max rel     ATOL      margin   model sqrt(K)');
console.log('  ---  -----  ----------  ----------  ----------  ---------  ------   ------------');

const rows = [];
let worstMargin = Infinity;
for (const K of K_VALUES) {
  const abs = [];
  const rel = [];
  for (let s = 1; s <= SEEDS; s++) {
    const { maxAbs, maxRel, relSamples } = runOne(K, s * 0x9e3779b9);
    abs.push(maxAbs);
    // Only record a relative sample when some element was actually above the
    // crossover; otherwise RTOL was untested at this K and a 0 would read as
    // "perfectly accurate" rather than "not measured".
    if (relSamples > 0) rel.push(maxRel);
  }
  abs.sort((x, y) => x - y);
  rel.sort((x, y) => x - y);
  const mx = abs[abs.length - 1];
  const atol = atolFor(K);
  const margin = atol / mx;
  worstMargin = Math.min(worstMargin, margin);
  // Normalised by sqrt(K): flat across K means the sqrt(K) model holds.
  const perSqrtK = mx / Math.sqrt(K);
  const relMax = rel.length ? rel[rel.length - 1] : null;
  rows.push({ K, mx, atol, margin, perSqrtK, relMax });
  console.log(
    `  ${String(K).padStart(3)}  ${String(SEEDS).padStart(5)}  ` +
      `${mx.toExponential(3)}  ${pct(abs, 0.5).toExponential(3)}  ` +
      `${(relMax === null ? '   n/a   ' : relMax.toExponential(3)).padStart(9)}  ${atol.toExponential(2)}  ` +
      `${margin.toFixed(1).padStart(5)}x   ${perSqrtK.toExponential(2)}`,
  );
}

console.log('');
console.log('VERDICT');
const allPass = rows.every((r) => r.margin > 1);
console.log(`  ATOL covers every seed at every K : ${allPass ? 'YES' : 'NO'}`);
console.log(`  worst margin observed            : ${worstMargin.toFixed(1)}x`);

// If error really grows like sqrt(K), max|abs|/sqrt(K) is constant across K and
// the ratio below is ~1. A large spread means the model is wrong even when the
// number happens to pass.
if (rows.length < 2) {
  // With a single K the ratio is trivially 1.00x, which would read as
  // confirmation of a claim this run cannot test at all.
  console.log('  sqrt(K) model: NOT TESTED — needs >1 K value (drop --quick)');
} else {
  const norm = rows.map((r) => r.perSqrtK);
  const spread = Math.max(...norm) / Math.min(...norm);
  console.log(`  sqrt(K) model: max/min of (max|abs| / sqrt(K)) = ${spread.toFixed(2)}x`);
  console.log(
    spread < 2
      ? '    -> flat across K: sqrt(K) scaling holds, ATOL will extrapolate'
      : '    -> NOT flat: sqrt(K) is the wrong exponent; ATOL happens to pass but the formula is wrong',
  );
}

console.log('');
console.log(`RTOL guidance (currently ${RTOL_CURRENT.toExponential(0)}), measured only where RTOL governs:`);
const relRows = rows.filter((r) => r.relMax !== null);
if (!relRows.length) {
  console.log('  no element exceeded the ATOL/RTOL crossover at any K —');
  console.log('  RTOL was never the deciding term, so this run says nothing about it.');
  process.exit(allPass ? 0 : 1);
}
const relWorst = Math.max(...relRows.map((r) => r.relMax));
const suggested = 10 * relWorst;
console.log(`  crossover |e| > ATOL/RTOL, e.g. ${relCrossover(64).toFixed(3)} at K=64`);
console.log(`  worst relative error above crossover     : ${relWorst.toExponential(3)}`);
console.log(`  10x headroom would allow RTOL            : ${suggested.toExponential(3)}`);
console.log(
  suggested < 1e-2
    ? '    -> 1e-2 is safe and 10x tighter than today, so a real kernel bug on\n' +
        '       large-magnitude elements would no longer hide inside the band.'
    : `    -> 1e-1 is about right; do NOT tighten below ${suggested.toExponential(1)}.`,
);

process.exit(allPass ? 0 : 1);
