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

console.log('matmul: OK');
