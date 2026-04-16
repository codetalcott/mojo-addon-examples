// spike/test-roundtrip.js — Day 1 smoke test.
//
// PASS on Day 1 means:
//   - spike/build/embed.node loads
//   - tokenize("hello world") produces non-empty Int32Array
//   - embedTokens() returns without throwing
//   - dst is filled with the stub value (all zeros) at the right shape
//
// PASS does NOT yet mean correctness. That's Day 2+ once _stub_forward
// is replaced with a real MAX graph pass and compared against ground-truth.bin.

const path = require('path');
const assert = require('assert');
const { tokenize } = require('./tokenize');

const EMBED_DIM = 384;

async function main() {
  const addon = require(path.join(__dirname, 'build', 'embed.node'));
  assert.ok(typeof addon.embedTokens === 'function', 'embedTokens export missing');

  const texts = ['hello world', 'the quick brown fox'];
  const { ids, mask, batch, seqLen } = await tokenize(texts);

  assert.strictEqual(batch, 2);
  assert.ok(seqLen > 0 && seqLen <= 128);
  assert.strictEqual(ids.length, batch * seqLen);
  assert.strictEqual(mask.length, batch * seqLen);

  const dst = new Float32Array(batch * EMBED_DIM);
  addon.embedTokens(ids, mask, batch, seqLen, dst);

  // Day 1: stub fills zeros. Assert the shape is right, not the values.
  assert.strictEqual(dst.length, batch * EMBED_DIM);
  const allZero = dst.every((v) => v === 0);
  console.log(`roundtrip: batch=${batch} seqLen=${seqLen} dst.length=${dst.length} allZero=${allZero}`);
  console.log('first 8 values of row 0:', Array.from(dst.slice(0, 8)));

  // Day 1 expectation: allZero === true (stub). Day 2+: replace with
  // cosine-similarity check vs fixtures/ground-truth.bin.
  if (!allZero) {
    console.warn('note: dst has non-zero values — you past Day 1 stub behavior.');
  }
  console.log('PASS');
}

main().catch((e) => {
  console.error('FAIL:', e);
  process.exit(1);
});
