const assert = require('node:assert');
const addon = require('./build/wyhash.node');

const buf = Buffer.from('hello');
const hash1 = addon.wyHash(buf);
const hash2 = addon.wyHash(buf);

assert.strictEqual(typeof hash1, 'bigint', 'returns BigInt');
assert.strictEqual(hash1, hash2, 'deterministic');

const hash64 = addon.wyHash64(buf);
assert.strictEqual(typeof hash64, 'number', 'wyHash64 returns Number');

// Seeded hash should differ from unseeded
const seeded = addon.wyHash(buf, 42n);
assert.notStrictEqual(seeded, hash1, 'seed changes output');

console.log('wyhash: OK');
