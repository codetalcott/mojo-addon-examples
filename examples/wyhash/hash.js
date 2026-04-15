// wyhash/hash.js — Correctness tests + benchmarks for wyhash addon

const addon = require('./build/wyhash.node');
const { performance } = require('perf_hooks');

// --- JS reference wyhash (BigInt arithmetic, slow but correct) ---

const _WYP0 = 0xa0761d6478bd642fn;
const _WYP1 = 0xe7037ed1a0b428dbn;
const _WYP2 = 0x8ebc6af09c88c6e3n;
const _WYP3 = 0x589965cc75374cc3n;
const MASK64 = (1n << 64n) - 1n;

function wymum(a, b) {
  const full = a * b;  // BigInt: arbitrary precision
  const lo = full & MASK64;
  const hi = (full >> 64n) & MASK64;
  return lo ^ hi;
}

function wyr8(buf, offset) {
  return buf.readBigUInt64LE(offset);
}

function wyr4(buf, offset) {
  return BigInt(buf.readUInt32LE(offset));
}

function wyr3(buf, k, len) {
  return (BigInt(buf[k]) << 16n) | (BigInt(buf[k + (len >> 1)]) << 8n) | BigInt(buf[k + len - 1]);
}

function jsWyhash(buf, seed = 0n) {
  seed = BigInt.asUintN(64, seed);
  const len = buf.length;
  seed ^= wymum(seed ^ _WYP0, _WYP1);
  seed = BigInt.asUintN(64, seed);
  let a = 0n, b = 0n;

  if (len <= 16) {
    if (len >= 4) {
      a = (wyr4(buf, 0) << 32n) | wyr4(buf, (len >> 3) << 2);
      b = (wyr4(buf, len - 4) << 32n) | wyr4(buf, len - ((len >> 3) << 2) - 4);
    } else if (len > 0) {
      a = wyr3(buf, 0, len);
      b = 0n;
    }
  } else if (len <= 48) {
    // 17-48 bytes: first 16 always safe, second 16 only if len > 32
    seed = BigInt.asUintN(64, wymum(wyr8(buf, 0) ^ _WYP1, wyr8(buf, 8) ^ seed));
    if (len > 32) {
      seed = BigInt.asUintN(64, wymum(wyr8(buf, 16) ^ _WYP2, wyr8(buf, 24) ^ seed));
    }
    a = wyr8(buf, len - 16);
    b = wyr8(buf, len - 8);
  } else {
    let see1 = seed;
    let see2 = seed;
    let i = 0;
    let remaining = len;
    while (remaining > 48) {
      seed = BigInt.asUintN(64, wymum(wyr8(buf, i) ^ _WYP1, wyr8(buf, i + 8) ^ seed));
      see1 = BigInt.asUintN(64, wymum(wyr8(buf, i + 16) ^ _WYP2, wyr8(buf, i + 24) ^ see1));
      see2 = BigInt.asUintN(64, wymum(wyr8(buf, i + 32) ^ _WYP3, wyr8(buf, i + 40) ^ see2));
      i += 48;
      remaining -= 48;
    }
    seed ^= see1 ^ see2;
    seed = BigInt.asUintN(64, seed);
    const tail = len - remaining;
    if (remaining > 32) {
      seed = BigInt.asUintN(64, wymum(wyr8(buf, tail) ^ _WYP1, wyr8(buf, tail + 8) ^ seed));
      see1 = BigInt.asUintN(64, wymum(wyr8(buf, tail + 16) ^ _WYP2, wyr8(buf, tail + 24) ^ see1));
      seed ^= see1;
      seed = BigInt.asUintN(64, seed);
    } else if (remaining > 16) {
      seed = BigInt.asUintN(64, wymum(wyr8(buf, tail) ^ _WYP1, wyr8(buf, tail + 8) ^ seed));
    }
    a = wyr8(buf, len - 16);
    b = wyr8(buf, len - 8);
  }

  a = BigInt.asUintN(64, a);
  b = BigInt.asUintN(64, b);
  return BigInt.asUintN(64, wymum(BigInt.asUintN(64, _WYP1 ^ BigInt(len)), wymum(a ^ _WYP1, b ^ seed)));
}

// --- JS FNV-1a hash (Number return, practical baseline) ---

function jsFnv1a(buf) {
  let hash = 2166136261;
  for (let i = 0; i < buf.length; i++) {
    hash ^= buf[i];
    hash = (hash * 16777619) | 0;
  }
  return hash >>> 0;
}

// --- Correctness tests ---

console.log('=== Correctness Tests ===\n');

const testCases = [
  { label: 'empty', buf: Buffer.from('') },
  { label: '"a"', buf: Buffer.from('a') },
  { label: '"ab"', buf: Buffer.from('ab') },
  { label: '"abc"', buf: Buffer.from('abc') },
  { label: '"hello"', buf: Buffer.from('hello') },
  { label: '"message digest"', buf: Buffer.from('message digest') },
  { label: 'alphabet (26 bytes)', buf: Buffer.from('abcdefghijklmnopqrstuvwxyz') },
  { label: '48 bytes', buf: Buffer.alloc(48, 0x42) },
  { label: '64 zeros', buf: Buffer.alloc(64) },
  { label: '100 bytes', buf: Buffer.alloc(100, 0xAB) },
];

let allPass = true;
for (const { label, buf } of testCases) {
  const mojoHash = addon.wyHash(buf);
  const jsHash = jsWyhash(buf);
  const pass = mojoHash === jsHash;
  if (!pass) allPass = false;
  console.log(`  ${label}: Mojo=0x${mojoHash.toString(16)} JS=0x${jsHash.toString(16)} ${pass ? 'PASS' : 'FAIL'}`);
}

// Test with seed
const seedBuf = Buffer.from('hello');
const mojoSeeded = addon.wyHash(seedBuf, 42n);
const jsSeeded = jsWyhash(seedBuf, 42n);
const seedPass = mojoSeeded === jsSeeded;
if (!seedPass) allPass = false;
console.log(`  seeded(42): Mojo=0x${mojoSeeded.toString(16)} JS=0x${jsSeeded.toString(16)} ${seedPass ? 'PASS' : 'FAIL'}`);

// Test wyHash64 returns a Number
const h64 = addon.wyHash64(Buffer.from('test'));
console.log(`  wyHash64 type: ${typeof h64} ${typeof h64 === 'number' ? 'PASS' : 'FAIL'}`);
if (typeof h64 !== 'number') allPass = false;

// Test Uint8Array input (not just Buffer)
const uint8Input = new Uint8Array([1, 2, 3, 4, 5]);
const mojoUint8 = addon.wyHash(uint8Input);
const jsUint8 = jsWyhash(Buffer.from(uint8Input));
const uint8Pass = mojoUint8 === jsUint8;
if (!uint8Pass) allPass = false;
console.log(`  Uint8Array: Mojo=0x${mojoUint8.toString(16)} JS=0x${jsUint8.toString(16)} ${uint8Pass ? 'PASS' : 'FAIL'}`);

console.log(`\n${allPass ? 'All correctness tests passed!' : 'SOME TESTS FAILED!'}\n`);

// --- Benchmarks ---

function bench(name, fn, warmup = 5, iters = 100) {
  for (let i = 0; i < warmup; i++) fn();
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) fn();
  const ms = (performance.now() - t0) / iters;
  return { name, ms };
}

function runBenchmarks(label, size, iters) {
  const buf = Buffer.alloc(size);
  // Fill with pseudo-random data
  for (let i = 0; i < size; i++) buf[i] = (i * 7 + 13) & 0xFF;

  console.log(`--- ${label} (${size >= 1048576 ? (size / 1048576) + 'MB' : size >= 1024 ? (size / 1024) + 'KB' : size + 'B'}) ---`);

  const jsR = bench('JS wyhash (BigInt)', () => jsWyhash(buf), 3, iters);
  const mojoR = bench('Mojo wyHash (BigInt)', () => addon.wyHash(buf), 5, iters);
  console.log(`  wyHash (BigInt):  JS ${jsR.ms.toFixed(3)}ms  Mojo ${mojoR.ms.toFixed(3)}ms  ${(jsR.ms / mojoR.ms).toFixed(1)}x`);

  const fnvR = bench('JS FNV-1a (Number)', () => jsFnv1a(buf), 3, iters);
  const mojo64R = bench('Mojo wyHash64 (Number)', () => addon.wyHash64(buf), 5, iters);
  console.log(`  wyHash64 (Number): JS FNV-1a ${fnvR.ms.toFixed(3)}ms  Mojo ${mojo64R.ms.toFixed(3)}ms  ${(fnvR.ms / mojo64R.ms).toFixed(1)}x`);

  console.log();
}

console.log('=== Benchmarks ===\n');
runBenchmarks('64 bytes', 64, 500000);
runBenchmarks('1 KB', 1024, 100000);
runBenchmarks('64 KB', 65536, 10000);
runBenchmarks('1 MB', 1048576, 500);
runBenchmarks('16 MB', 16777216, 50);
