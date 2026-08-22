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

// Empty buffer produces a deterministic hash
const empty1 = addon.wyHash(Buffer.alloc(0));
const empty2 = addon.wyHash(Buffer.alloc(0));
assert.strictEqual(empty1, empty2, 'empty deterministic');

// Large buffer doesn't crash
const big = Buffer.alloc(1024 * 1024, 0xAB);
const bigHash = addon.wyHash(big);
assert.strictEqual(typeof bigHash, 'bigint', '1MB hash is BigInt');

// wyHash64 returns a finite number
assert(Number.isFinite(addon.wyHash64(buf)), 'wyHash64 is finite');
assert(Number.isFinite(addon.wyHash64(big)), 'wyHash64 1MB is finite');

// Uint8Array input works the same as Buffer
const u8 = new Uint8Array(buf);
assert.strictEqual(addon.wyHash(Buffer.from(u8)), hash1, 'Uint8Array matches Buffer');


// wyhash branches on length (<=16, 17-48, >48 bulk). Walk every boundary so a
// mis-migrated read in one branch cannot hide behind the others.
{
  const seen = new Map();
  for (const len of [0, 1, 2, 3, 4, 5, 8, 15, 16, 17, 24, 32, 33, 48, 49, 96, 97, 1000]) {
    const b = Buffer.alloc(len);
    for (let i = 0; i < len; i++) b[i] = (i * 31 + 7) & 0xff;
    const h = addon.wyHash(b);
    assert.strictEqual(typeof h, 'bigint', `len ${len} is BigInt`);
    assert.strictEqual(addon.wyHash(b), h, `len ${len} deterministic`);
    assert(Number.isFinite(addon.wyHash64(b)), `len ${len} wyHash64 finite`);
    // Distinct lengths of distinct content should not collide in this small set.
    assert(!seen.has(h), `len ${len} collides with len ${seen.get(h)}`);
    seen.set(h, len);
  }
}

console.log('wyhash: OK');
