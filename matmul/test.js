const assert = require('node:assert');
const addon = require('./build/matmul.node');

// 2x2 identity multiply: [[1,2],[3,4]] * [[1,0],[0,1]] = [[1,2],[3,4]]
const a = new Float64Array([1, 2, 3, 4]);
const b = new Float64Array([1, 0, 0, 1]);
const c = new Float64Array(4);

addon.matmulNaive(a, b, c, 2, 2, 2);
assert.deepStrictEqual(Array.from(c), [1, 2, 3, 4], 'matmulNaive identity');

addon.matmulVectorized(a, b, c, 2, 2, 2);
assert.deepStrictEqual(Array.from(c), [1, 2, 3, 4], 'matmulVectorized identity');

addon.matmulParallel(a, b, c, 2, 2, 2);
assert.deepStrictEqual(Array.from(c), [1, 2, 3, 4], 'matmulParallel identity');

console.log('matmul: OK');
