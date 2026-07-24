// packages/embed/demo.js — end-to-end demo.
//
// @qkstat/embed (MAX on H100 via Python interop) embeds 1000 docs,
// @qkstat/retrieve's GpuIndex does exact cosine search, queries run,
// accuracy + latency reported.

const path = require('path');
const { tokenize } = require('./tokenize');
const { buildCorpus, buildQueries } = require('./corpus');
const { GpuIndex } = require('../retrieve/lib/GpuIndex');

const EMBED_DIM = 384;
const K = 10;

async function embedBatch(addon, docs) {
  const { ids, mask, batch, seqLen } = await tokenize(docs);
  const dst = new Float32Array(batch * EMBED_DIM);
  addon.embedTokens(ids, mask, batch, seqLen, dst);
  return { embeddings: dst, batch, seqLen };
}

async function main() {
  const addon = require(path.join(__dirname, 'build', 'embed.node'));
  if (typeof addon.embedTokens !== 'function') {
    throw new Error('embed.node missing embedTokens — run packages/embed/build.sh');
  }

  console.log('building corpus (5 clusters × 200 = 1000 docs)...');
  const { docs, clusters, clusterNames } = buildCorpus(200);

  console.log('\nembedding corpus in batches of 64...');
  const BATCH = 64;
  const embeddings = new Float32Array(docs.length * EMBED_DIM);
  const tEmbedStart = performance.now();
  let firstBatchLogged = false;
  for (let off = 0; off < docs.length; off += BATCH) {
    const batchDocs = docs.slice(off, off + BATCH);
    const { embeddings: e, batch, seqLen } = await embedBatch(addon, batchDocs);
    embeddings.set(e, off * EMBED_DIM);
    if (!firstBatchLogged) {
      console.log(`  first batch: batch=${batch}, seq_len=${seqLen}`);
      firstBatchLogged = true;
    }
  }
  const embedMs = performance.now() - tEmbedStart;
  console.log(`corpus embed time: ${embedMs.toFixed(0)}ms  (${(docs.length * 1000 / embedMs).toFixed(0)} docs/sec)`);

  console.log('\nbuilding GpuIndex (corpus → separate CUDA context for packages/retrieve)...');
  const tIdxStart = performance.now();
  const index = new GpuIndex({ docs, embeddings, dim: EMBED_DIM });
  console.log(`GpuIndex build: ${(performance.now() - tIdxStart).toFixed(1)}ms`);

  const { queries, queryClusterTruth } = buildQueries(2);
  console.log(`\nrunning ${queries.length} queries (k=${K})...`);

  // Warmup
  {
    const { embeddings: w } = await embedBatch(addon, [queries[0]]);
    index.search(w, K);
  }

  const embedLatencies = [];
  const searchLatencies = [];
  const queryLatencies = [];
  let totalHits = 0;

  for (let qi = 0; qi < queries.length; qi++) {
    const query = queries[qi];
    const expectedCluster = queryClusterTruth[qi];

    const t0 = performance.now();
    const { embeddings: qEmb } = await embedBatch(addon, [query]);
    const tEmb = performance.now();
    const results = index.search(qEmb, K);
    const tSearch = performance.now();

    embedLatencies.push(tEmb - t0);
    searchLatencies.push(tSearch - tEmb);
    queryLatencies.push(tSearch - t0);

    const hits = results.filter(r => clusters[r.index] === expectedCluster).length;
    totalHits += hits;

    if (qi < 3) {
      console.log(`\n  Q${qi} [${clusterNames[expectedCluster]}]: "${query}"`);
      console.log(`    hits=${hits}/${K}  embed=${(tEmb - t0).toFixed(2)}ms  search=${(tSearch - tEmb).toFixed(3)}ms`);
      for (let i = 0; i < 3; i++) {
        const r = results[i];
        const col = clusterNames[clusters[r.index]];
        console.log(`    ${i + 1}. [${col}] score=${r.score.toFixed(4)} :: ${r.doc.slice(0, 80)}`);
      }
    }
  }

  // Summary
  const avgL = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const p50 = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length * 0.5)];
  const pct = (100 * totalHits / (queries.length * K)).toFixed(1);

  console.log('\n==========================');
  console.log('SUMMARY — @qkstat/embed (MAX on H100 + @qkstat/retrieve exact search)');
  console.log('==========================');
  console.log(`corpus: ${docs.length} docs, ${clusterNames.length} clusters`);
  console.log(`corpus embed throughput: ${(docs.length * 1000 / embedMs).toFixed(0)} docs/sec  (${embedMs.toFixed(0)}ms total)`);
  console.log(`in-cluster hits: ${totalHits}/${queries.length * K} (${pct}%)`);
  console.log(`per-query latency (embed):   avg=${avgL(embedLatencies).toFixed(2)}ms   p50=${p50(embedLatencies).toFixed(2)}ms`);
  console.log(`per-query latency (search):  avg=${avgL(searchLatencies).toFixed(3)}ms   p50=${p50(searchLatencies).toFixed(3)}ms`);
  console.log(`per-query latency (total):   avg=${avgL(queryLatencies).toFixed(2)}ms   p50=${p50(queryLatencies).toFixed(2)}ms`);

  index.close();
  console.log('\nDONE');
}

main().catch((e) => { console.error('FAIL:', e); process.exit(1); });
