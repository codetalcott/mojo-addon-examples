// packages/embed/test-async.js — thread safety of the addon's Python calls.
//
// embedTokensAsync runs on the libuv threadpool and embedTokens on the JS
// thread, and both call into CPython. Two bugs lived here until 2026-10:
//
//   - Mojo's Python interop leaves the GIL with whichever thread initialized
//     CPython, so a call from any other thread segfaulted (NULL thread state,
//     fault at 0x10). The async path died on its second call.
//   - embed.py built its engine without a lock, so concurrent first calls each
//     built one, and building several at once corrupted the heap.
//
// Both kill the process instead of failing an assertion, so a crash (exit 134
// or 139) is this test's failure mode as much as a mismatch is.
//
// Order matters: the concurrent round must be the FIRST Python call in the
// process, or the engine already exists and the lock goes untested.
//
// PASS: every call, on every thread, returns the same embeddings as the first.
//
// Run: pixi run node packages/embed/test-async.js

const path = require('path');
const { tokenize } = require('./tokenize');

const EMBED_DIM = 384;
const CONCURRENCY = 8;  // > libuv's default pool of 4, so workers contend
const TOL = 1e-5;

async function main() {
  const addon = require(path.join(__dirname, 'build', 'embed.node'));
  const { ids, mask, batch, seqLen } = await tokenize([
    'Comments explain why, not what.',
    'the gil belongs to one thread at a time',
  ]);
  const n = batch * EMBED_DIM;
  const sync = () => {
    const dst = new Float32Array(n);
    addon.embedTokens(ids, mask, batch, seqLen, dst);
    return dst;
  };
  const async1 = async () => {
    const dst = new Float32Array(n);
    await addon.embedTokensAsync(ids, mask, batch, seqLen, dst);
    return dst;
  };
  const concurrent = () => Promise.all(Array.from({ length: CONCURRENCY }, async1));

  let ref = null;
  let calls = 0;
  const check = (label, dst) => {
    calls++;
    for (let r = 0; r < batch; r++) {
      let sq = 0;
      for (let d = 0; d < EMBED_DIM; d++) sq += dst[r * EMBED_DIM + d] ** 2;
      if (Math.abs(Math.sqrt(sq) - 1) > 1e-3) {
        throw new Error(`${label}: row ${r} has L2 norm ${Math.sqrt(sq)}, expected 1 (output not written?)`);
      }
    }
    if (!ref) { ref = dst; return; }
    let worst = 0;
    for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(dst[i] - ref[i]));
    if (worst > TOL) throw new Error(`${label}: max |diff| ${worst} vs the first result`);
  };

  console.log(`cold: ${CONCURRENCY} concurrent embedTokensAsync as the first Python call...`);
  (await concurrent()).forEach((d) => check('cold concurrent', d));
  console.log('sync embedTokens on the JS thread, after the workers...');
  check('sync after async', sync());
  console.log('sequential embedTokensAsync, landing on different workers...');
  for (let i = 0; i < 4; i++) check(`sequential async #${i + 1}`, await async1());
  console.log('interleaved sync and concurrent async...');
  for (let i = 0; i < 3; i++) {
    check(`interleaved sync #${i + 1}`, sync());
    (await concurrent()).forEach((d) => check(`interleaved concurrent #${i + 1}`, d));
  }

  console.log(`\n✅ async PASS: ${calls} calls matched within ${TOL}`);
}

main().catch((e) => {
  console.error('FAIL:', e);
  process.exit(1);
});
