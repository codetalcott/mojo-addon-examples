// simd-search/search.js — SIMD byte search benchmark
//
// Demonstrates Mojo's SIMD byte-level pattern matching vs pure JavaScript.
// Operations impossible to optimize in JS: countByte, countLines, searchAll.
//
// Build:  pixi run bash simd-search/build.sh
// Run:    node simd-search/search.js

const addon = require('./build/search.node');

// --- Pure JS baselines -------------------------------------------------------

function jsCountByte(buf, byte) {
  let count = 0;
  for (let i = 0; i < buf.length; i++) {
    if (buf[i] === byte) count++;
  }
  return count;
}

function jsCountLines(buf) {
  return jsCountByte(buf, 0x0A);
}

function jsSearchAll(buf, needle) {
  const positions = [];
  for (let i = 0; i <= buf.length - needle.length; i++) {
    let found = true;
    for (let k = 0; k < needle.length; k++) {
      if (buf[i + k] !== needle[k]) { found = false; break; }
    }
    if (found) positions.push(i);
  }
  return new Uint32Array(positions);
}

// --- Correctness checks ------------------------------------------------------

console.log('=== Correctness ===\n');

// countByte
const testBuf = Buffer.from('hello world hello');
const hCount = addon.countByte(testBuf, 'l'.charCodeAt(0));
console.log(`  countByte('hello world hello', 'l'): ${hCount} (expected 5) ${hCount === 5 ? 'PASS' : 'FAIL'}`);

// countLines
const linesBuf = Buffer.from('line1\nline2\nline3\n');
const lineCount = addon.countLines(linesBuf);
console.log(`  countLines('line1\\nline2\\nline3\\n'): ${lineCount} (expected 3) ${lineCount === 3 ? 'PASS' : 'FAIL'}`);

// searchAll — single byte
const positions1 = addon.searchAll(testBuf, Buffer.from('l'));
const expected1 = [2, 3, 9, 14, 15];
const pass1 = positions1.length === expected1.length && expected1.every((v, i) => positions1[i] === v);
console.log(`  searchAll single-byte: [${Array.from(positions1)}] (expected [${expected1}]) ${pass1 ? 'PASS' : 'FAIL'}`);

// searchAll — multi-byte
const positions2 = addon.searchAll(testBuf, Buffer.from('hello'));
const expected2 = [0, 12];
const pass2 = positions2.length === expected2.length && expected2.every((v, i) => positions2[i] === v);
console.log(`  searchAll multi-byte: [${Array.from(positions2)}] (expected [${expected2}]) ${pass2 ? 'PASS' : 'FAIL'}`);

// searchAll — no match
const positions3 = addon.searchAll(testBuf, Buffer.from('xyz'));
console.log(`  searchAll no match: length=${positions3.length} (expected 0) ${positions3.length === 0 ? 'PASS' : 'FAIL'}`);

// Uint8Array input (not just Buffer)
const u8 = new Uint8Array([65, 66, 65, 67, 65]);
const aCount = addon.countByte(u8, 65);
console.log(`  countByte(Uint8Array, 65): ${aCount} (expected 3) ${aCount === 3 ? 'PASS' : 'FAIL'}`);

// --- Benchmark ---------------------------------------------------------------

function bench(name, fn, iters) {
  for (let i = 0; i < Math.min(iters, 50); i++) fn(); // warmup
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

function formatSize(bytes) {
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(0) + 'MB';
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(0) + 'KB';
  return bytes + 'B';
}

// Generate test buffer with ~1% newlines
function makeBuffer(size) {
  const buf = Buffer.alloc(size);
  for (let i = 0; i < size; i++) {
    buf[i] = Math.random() < 0.01 ? 0x0A : Math.floor(Math.random() * 255) + 1;
  }
  return buf;
}

const sizes = [
  [1024, 100000],
  [65536, 10000],
  [1048576, 1000],
  [16777216, 50],
  [104857600, 5],
];

console.log('\n--- countByte benchmark ---');

for (const [SIZE, ITERS] of sizes) {
  console.log(`\n=== countByte: ${formatSize(SIZE)} buffer ===\n`);
  const buf = makeBuffer(SIZE);
  const target = 0x0A;

  const jsResult = bench('JS loop  ', () => jsCountByte(buf, target), ITERS);
  const mojoResult = bench('Mojo SIMD', () => addon.countByte(buf, target), ITERS);

  const speedup = (jsResult.ms / mojoResult.ms).toFixed(1);
  console.log(`  ${jsResult.name}: ${jsResult.ms.toFixed(1)}ms  ${formatOps(jsResult.opsPerSec)} ops/sec  (baseline)`);
  console.log(`  ${mojoResult.name}: ${mojoResult.ms.toFixed(1)}ms  ${formatOps(mojoResult.opsPerSec)} ops/sec  ${speedup}x`);

  if (typeof addon.countByteGpu === 'function') {
    try {
      // Correctness spot-check.
      const cpuSpot = addon.countByte(buf, target);
      const gpuSpot = addon.countByteGpu(buf, target);
      if (cpuSpot !== gpuSpot) {
        throw new Error(`mismatch: cpu=${cpuSpot} gpu=${gpuSpot}`);
      }
      const gpuResult = bench('Mojo GPU ', () => addon.countByteGpu(buf, target), ITERS);
      const gSpeed = (jsResult.ms / gpuResult.ms).toFixed(1);
      console.log(`  ${gpuResult.name}: ${gpuResult.ms.toFixed(1)}ms  ${formatOps(gpuResult.opsPerSec)} ops/sec  ${gSpeed}x`);
    } catch (e) {
      console.log(`  Mojo GPU : skipped (${e.message})`);
    }
  }
}

console.log('\n--- searchAll benchmark (single-byte, ~1% hit rate) ---');

for (const [SIZE, ITERS] of sizes.slice(0, 3)) {
  console.log(`\n=== searchAll: ${formatSize(SIZE)} buffer ===\n`);
  const buf = makeBuffer(SIZE);
  const needle = Buffer.from([0x0A]);

  const jsResult = bench('JS loop  ', () => jsSearchAll(buf, needle), ITERS);
  const mojoResult = bench('Mojo SIMD', () => addon.searchAll(buf, needle), ITERS);

  const speedup = (jsResult.ms / mojoResult.ms).toFixed(1);
  console.log(`  ${jsResult.name}: ${jsResult.ms.toFixed(1)}ms  ${formatOps(jsResult.opsPerSec)} ops/sec  (baseline)`);
  console.log(`  ${mojoResult.name}: ${mojoResult.ms.toFixed(1)}ms  ${formatOps(mojoResult.opsPerSec)} ops/sec  ${speedup}x`);
}

console.log('\n--- searchAll benchmark (multi-byte "http") ---');

for (const [SIZE, ITERS] of sizes.slice(0, 3)) {
  console.log(`\n=== searchAll "http": ${formatSize(SIZE)} buffer ===\n`);
  const buf = makeBuffer(SIZE);
  // Inject some "http" strings at random positions
  const httpBuf = Buffer.from('http');
  for (let i = 0; i < SIZE / 1000; i++) {
    const pos = Math.floor(Math.random() * (SIZE - 4));
    httpBuf.copy(buf, pos);
  }
  const needle = Buffer.from('http');

  const jsResult = bench('JS loop  ', () => jsSearchAll(buf, needle), ITERS);
  const mojoResult = bench('Mojo SIMD', () => addon.searchAll(buf, needle), ITERS);

  const speedup = (jsResult.ms / mojoResult.ms).toFixed(1);
  console.log(`  ${jsResult.name}: ${jsResult.ms.toFixed(1)}ms  ${formatOps(jsResult.opsPerSec)} ops/sec  (baseline)`);
  console.log(`  ${mojoResult.name}: ${mojoResult.ms.toFixed(1)}ms  ${formatOps(mojoResult.opsPerSec)} ops/sec  ${speedup}x`);
}
