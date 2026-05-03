// packages/embed/test-pipeline.js — RagPipeline integration test.
//
// Exercises the embed→GpuIndex orchestration: warmup idempotence, addTexts
// rebuild, search lifecycle, batched search, close-then-search rejection.
// Uses a deterministic bag-of-words mock engine by default so the test runs
// anywhere (M4 cannot run the real MAX graph; that path is covered by
// packages/embed/demo.js + bench.js on RunPod H100).
//
// Set EMBED_REAL=1 to swap in the real EmbeddingEngine (RunPod only).
//
// Run: pixi run node packages/embed/test-pipeline.js

const assert = require('assert');
const { RagPipeline, EmbeddingEngine } = require('./');

const DOCS = [
  'authentication and login session management',
  'database schema migration and rollback',
  'http server routing middleware patterns',
  'image processing convolution kernels',
  'gpu memory allocation cuda streams',
  'json parsing recursive descent grammar',
  'merge sort divide and conquer algorithm',
  'tcp congestion control window scaling',
  'vector embedding cosine similarity search',
  'yaml configuration file schema validation',
];

const QUERIES = [
  { text: 'authentication login session', expectIdx: 0 },
  { text: 'schema migration rollback', expectIdx: 1 },
  { text: 'gpu cuda streams', expectIdx: 4 },
  { text: 'cosine embedding vector', expectIdx: 8 },
];

// Deterministic bag-of-words mock — maps each unique token to a fixed basis
// vector and L2-normalizes the mean. Exercises GpuIndex (real matmul + top-k)
// without depending on MAX/Python. Token-overlap drives the expected top-1.
class MockEngine {
  constructor(dim = 384) {
    this.dim = dim;
    this._warmed = false;
    this._basis = new Map();
  }
  async warmup() { this._warmed = true; }
  _vec(token) {
    let v = this._basis.get(token);
    if (!v) {
      v = new Float32Array(this.dim);
      // Stable hash → fill the vector with a sin-pattern keyed by the token
      let h = 0;
      for (let i = 0; i < token.length; i++) h = (h * 31 + token.charCodeAt(i)) | 0;
      for (let i = 0; i < this.dim; i++) v[i] = Math.sin((h ^ (i * 2654435761)) >>> 0);
      this._basis.set(token, v);
    }
    return v;
  }
  async embedAsync(texts) {
    const out = new Float32Array(texts.length * this.dim);
    for (let i = 0; i < texts.length; i++) {
      const tokens = texts[i].toLowerCase().split(/\s+/).filter(Boolean);
      const acc = new Float32Array(this.dim);
      for (const t of tokens) {
        const v = this._vec(t);
        for (let d = 0; d < this.dim; d++) acc[d] += v[d];
      }
      let n = 0;
      for (let d = 0; d < this.dim; d++) n += acc[d] * acc[d];
      n = 1.0 / (Math.sqrt(n) + 1e-12);
      for (let d = 0; d < this.dim; d++) out[i * this.dim + d] = acc[d] * n;
    }
    return out;
  }
}

async function main() {
  const useReal = process.env.EMBED_REAL === '1';
  const engine = useReal ? new EmbeddingEngine() : new MockEngine();
  console.log(`engine: ${useReal ? 'REAL EmbeddingEngine (MAX)' : 'MockEngine (bag-of-words)'}`);

  console.log(`building RagPipeline (corpus = ${DOCS.length} docs)...`);
  const pipe = new RagPipeline({ engine });
  assert.strictEqual(pipe.size, 0, 'size must be 0 before addTexts');

  console.log('warmup #1...');
  const t0 = performance.now();
  await pipe.warmup();
  const w1 = performance.now() - t0;
  console.log(`  ${w1.toFixed(0)}ms`);

  console.log('warmup #2 (must be no-op)...');
  const t1 = performance.now();
  await pipe.warmup();
  const w2 = performance.now() - t1;
  console.log(`  ${w2.toFixed(2)}ms`);
  assert.ok(w2 < 5, `second warmup took ${w2}ms, expected <5ms (no-op)`);

  console.log('addTexts...');
  const { count, embedMs } = await pipe.addTexts(DOCS);
  assert.strictEqual(count, DOCS.length);
  assert.strictEqual(pipe.size, DOCS.length);
  console.log(`  embedded ${count} docs in ${embedMs.toFixed(0)}ms`);

  console.log(`running ${QUERIES.length} single-query searches...`);
  for (const q of QUERIES) {
    const hits = await pipe.search(q.text, 5);
    assert.ok(Array.isArray(hits), 'hits must be an array');
    assert.strictEqual(hits.length, 5);
    const top = hits[0];
    assert.ok(typeof top.doc === 'string');
    assert.ok(typeof top.score === 'number');
    assert.ok(typeof top.index === 'number');
    const ok = top.index === q.expectIdx;
    console.log(`  "${q.text}" → top1=${top.index} (${ok ? 'OK' : 'EXPECTED ' + q.expectIdx}) score=${top.score.toFixed(4)}`);
    assert.strictEqual(top.index, q.expectIdx, `top-1 mismatch for "${q.text}"`);
  }

  console.log('searchTexts (batched)...');
  const batched = await pipe.searchTexts(QUERIES.map(q => q.text), 3);
  assert.strictEqual(batched.length, QUERIES.length);
  for (let i = 0; i < QUERIES.length; i++) {
    assert.strictEqual(batched[i].length, 3);
    assert.strictEqual(batched[i][0].index, QUERIES[i].expectIdx, `batched top-1 mismatch for query ${i}`);
  }
  console.log(`  ${batched.length} queries × k=3 returned, top-1 matches single-query path`);

  console.log('addTexts again (rebuild path)...');
  await pipe.addTexts(DOCS.slice(0, 5));
  assert.strictEqual(pipe.size, 5);
  const hits2 = await pipe.search('authentication login session', 3);
  assert.strictEqual(hits2[0].index, 0);
  console.log('  rebuilt to 5 docs, search still works');

  console.log('close, then search must throw...');
  pipe.close();
  assert.strictEqual(pipe.size, 0);
  await assert.rejects(() => pipe.search('anything', 5), /no index/i);
  pipe.close();  // idempotent
  console.log('  ✓ search rejected with clear error; close() is idempotent');

  console.log('\n✅ RagPipeline test PASS');
}

main().catch((e) => {
  console.error('\n❌ FAIL:', e);
  process.exit(1);
});
