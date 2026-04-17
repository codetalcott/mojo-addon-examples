// packages/embed/test-roundtrip.js — end-to-end correctness test.
//
// Flow: tokenize sanity set in JS (WordPiece via @huggingface/transformers)
//   → call embedTokens() N-API → Mojo → Python (embed.py) → MAX on GPU
//   → embeddings back to JS → compare cosine similarity against
//     fixtures/ground-truth.bin from packages/embed/reference.js.
//
// PASS (Gate F4): cosine similarity ≥ 0.995 for every sentence in the set.

const path = require('path');
const fs = require('fs');
const assert = require('assert');
const { tokenize } = require('./tokenize');

const EMBED_DIM = 384;
const MIN_COSINE = 0.995;  // Gate F4 threshold

function loadGroundTruth() {
  const p = path.join(__dirname, 'fixtures', 'ground-truth.bin');
  if (!fs.existsSync(p)) {
    throw new Error(
      `${p} missing — run: node packages/embed/reference.js (once, on M4 is fine — CPU)`,
    );
  }
  const buf = fs.readFileSync(p);
  if (buf.slice(0, 4).toString('ascii') !== 'GRT1') {
    throw new Error(`${p}: bad magic (expected GRT1)`);
  }
  const dtype = buf.readUInt32LE(4);
  const n = buf.readUInt32LE(8);
  const dim = buf.readUInt32LE(12);
  if (dtype !== 0) throw new Error(`${p}: unsupported dtype=${dtype} (need float32)`);
  if (dim !== EMBED_DIM) throw new Error(`${p}: dim=${dim} != ${EMBED_DIM}`);
  const expected = 16 + n * dim * 4;
  if (buf.length !== expected) {
    throw new Error(`${p}: size mismatch (header says ${expected}, got ${buf.length})`);
  }
  const out = new Float32Array(n * dim);
  Buffer.from(out.buffer).set(buf.slice(16));
  return { n, dim, embeddings: out };
}

function loadSanitySet() {
  const p = path.join(__dirname, 'fixtures', 'sanity-set.txt');
  if (!fs.existsSync(p)) throw new Error(`${p} missing — run packages/embed/reference.js`);
  return fs.readFileSync(p, 'utf8').split('\n').filter(Boolean);
}

function cosine(a, b, offA, offB, dim) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < dim; i++) {
    const x = a[offA + i], y = b[offB + i];
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

async function main() {
  const addon = require(path.join(__dirname, 'build', 'embed.node'));
  assert.ok(typeof addon.embedTokens === 'function', 'embedTokens export missing');

  const sentences = loadSanitySet();
  const gt = loadGroundTruth();
  console.log(`sanity set: ${sentences.length} sentences, ground-truth shape: (${gt.n}, ${gt.dim})`);
  assert.strictEqual(sentences.length, gt.n, 'sanity-set.txt and ground-truth.bin disagree on N');

  console.log('tokenizing...');
  const { ids, mask, batch, seqLen } = await tokenize(sentences);
  console.log(`  tokens shape: [${batch}, ${seqLen}]  (batch=${batch}, seq_len=${seqLen})`);

  const dst = new Float32Array(batch * EMBED_DIM);
  console.log('calling embedTokens() — this will spin up MAX on first call (expect ~10-30s cold start)');
  const t0 = Date.now();
  addon.embedTokens(ids, mask, batch, seqLen, dst);
  const coldMs = Date.now() - t0;
  console.log(`  first-call latency: ${coldMs}ms (includes model load + compile)`);

  // Correctness check
  let minCos = 1.0, sumCos = 0, worstIdx = -1;
  for (let i = 0; i < batch; i++) {
    const c = cosine(dst, gt.embeddings, i * EMBED_DIM, i * EMBED_DIM, EMBED_DIM);
    sumCos += c;
    if (c < minCos) { minCos = c; worstIdx = i; }
  }
  const meanCos = sumCos / batch;
  console.log(`correctness (vs @huggingface/transformers CPU reference):`);
  console.log(`  mean cosine: ${meanCos.toFixed(6)}`);
  console.log(`  min cosine:  ${minCos.toFixed(6)}  (sentence ${worstIdx}: "${sentences[worstIdx]}")`);

  // Warm timing
  console.log('\nwarming up (5 iters)...');
  for (let i = 0; i < 5; i++) addon.embedTokens(ids, mask, batch, seqLen, dst);
  const ITERS = 20;
  const t1 = performance.now();
  for (let i = 0; i < ITERS; i++) addon.embedTokens(ids, mask, batch, seqLen, dst);
  const warmMs = (performance.now() - t1) / ITERS;
  console.log(`warm latency (batch-${batch}, seq_len-${seqLen}): ${warmMs.toFixed(2)}ms/op`);

  // Verdict
  if (minCos < MIN_COSINE) {
    console.error(`\n❌ GATE F4 FAIL: min cosine ${minCos.toFixed(4)} < ${MIN_COSINE}`);
    console.error(`First 8 values of embedding 0:`);
    console.error(`  ours:   [${Array.from(dst.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);
    console.error(`  truth:  [${Array.from(gt.embeddings.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);
    process.exit(1);
  }
  console.log(`\n✅ GATE F4 PASS: min cosine ${minCos.toFixed(4)} ≥ ${MIN_COSINE}`);
  console.log(`   (cold: ${coldMs}ms, warm: ${warmMs.toFixed(2)}ms/op over ${ITERS} iters)`);
}

main().catch((e) => {
  console.error('FAIL:', e);
  process.exit(1);
});
