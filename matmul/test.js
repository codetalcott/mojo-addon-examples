const assert = require('node:assert');
const addon = require('./build/matmul.node');

const variants = ['matmulNaive', 'matmulVectorized', 'matmulTiled', 'matmulParallel'];

// 2x2 identity multiply: [[1,2],[3,4]] * [[1,0],[0,1]] = [[1,2],[3,4]]
for (const fn of variants) {
  const c = new Float64Array(4);
  addon[fn](new Float64Array([1, 2, 3, 4]), new Float64Array([1, 0, 0, 1]), c, 2, 2, 2);
  assert.deepStrictEqual(Array.from(c), [1, 2, 3, 4], `${fn} identity`);
}

// 1x1 scalar multiply
for (const fn of variants) {
  const c = new Float64Array(1);
  addon[fn](new Float64Array([3]), new Float64Array([7]), c, 1, 1, 1);
  assert.strictEqual(c[0], 21, `${fn} 1x1`);
}

// Zero matrix: anything * zeros = zeros
for (const fn of variants) {
  const c = new Float64Array(4);
  addon[fn](new Float64Array([1, 2, 3, 4]), new Float64Array([0, 0, 0, 0]), c, 2, 2, 2);
  assert.deepStrictEqual(Array.from(c), [0, 0, 0, 0], `${fn} zero matrix`);
}


// Parallel/tiled path: sizes below cross TILE_SIZE (64) and the worker row
// stride, which the tiny cases above never reach. Checked against a plain JS
// reference so a miscomputed tile shows up as a value mismatch, not a crash.

// Deterministic PRNG so failures reproduce exactly.
function lcg(seed) {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return s / 4294967296; };
}

{
  const rnd = lcg(7);
  const ref = (a, b, M, K, N) => {
    const c = new Float64Array(M * N);
    for (let i = 0; i < M; i++)
      for (let j = 0; j < N; j++) {
        let s = 0;
        for (let p = 0; p < K; p++) s += a[i * K + p] * b[p * N + j];
        c[i * N + j] = s;
      }
    return c;
  };
  for (const [M, K, N] of [[96, 96, 96], [70, 130, 65]]) {
    const a = Float64Array.from({ length: M * K }, rnd);
    const b = Float64Array.from({ length: K * N }, rnd);
    const expect = ref(a, b, M, K, N);
    for (const fn of variants) {
      const c = new Float64Array(M * N);
      addon[fn](a, b, c, M, K, N);
      for (let i = 0; i < M * N; i++) {
        assert(Math.abs(c[i] - expect[i]) < 1e-9, `${fn} ${M}x${K}x${N} at ${i}`);
      }
    }
  }
}

console.log('matmul: OK');
