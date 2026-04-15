#!/usr/bin/env node
// Build the MS-MARCO real-embedding fixture used by examples/matmul/matmul_rag.js
// when invoked with --fixture=msmarco-10k.
//
// Pulls the first N passages and Q queries from BeIR's MS-MARCO via the
// HuggingFace datasets-server REST API (paginated, 100 rows/request — for
// N=10000 that's 100 small requests, no multi-GB download). Embeds with
// Xenova/all-MiniLM-L6-v2 (d=384, mean-pooled, L2-normalized) and writes
// two .bin files:
//
//   magic   : "MMR1" (4 bytes ASCII)
//   dtype   : uint32 little-endian (0 = float32)
//   d       : uint32 little-endian
//   n       : uint32 little-endian
//   data    : float32[n*d] little-endian, row-major (one row per vector)
//
// Usage:
//   node scripts/build-msmarco-fixture.js [--n=10000] [--queries=200] \
//     [--out=examples/rag-demo/fixtures]
//
// Caches raw passage/query JSON under <out>/.cache/ and ~25 MB of model
// weights to ~/.cache/huggingface/. Subsequent runs reuse both.

const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const argVal = (k, def) => {
  const m = args.find((a) => a.startsWith(`--${k}=`));
  return m ? m.split('=')[1] : def;
};

const N = parseInt(argVal('n', '10000'), 10);
const Q = parseInt(argVal('queries', '200'), 10);
const OUT_DIR = path.resolve(__dirname, '..', argVal('out', 'examples/rag-demo/fixtures'));
const CACHE_DIR = path.join(OUT_DIR, '.cache');

const DATASET = 'BeIR/msmarco';
const ROWS_API = 'https://datasets-server.huggingface.co/rows';
const PAGE = 100;
const MODEL = 'Xenova/all-MiniLM-L6-v2';
const D = 384;
const BATCH = 32;
const MAGIC = Buffer.from('MMR1', 'ascii');

function fixtureName(n) {
  if (n >= 1_000_000 && n % 1_000_000 === 0) return `msmarco-${n / 1_000_000}m`;
  if (n >= 1000 && n % 1000 === 0) return `msmarco-${n / 1000}k`;
  return `msmarco-${n}`;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function fetchPage(config, offset, length) {
  const url = `${ROWS_API}?dataset=${encodeURIComponent(DATASET)}&config=${config}&split=${config}&offset=${offset}&length=${length}`;
  // Up to 10 retries with capped exponential backoff. The HF datasets-server
  // routinely 429s under sustained polling; 60s is enough to clear the bucket.
  for (let attempt = 0; attempt < 10; attempt++) {
    const res = await fetch(url);
    if (res.ok) return res.json();
    if (res.status !== 429 && res.status < 500) {
      throw new Error(`HF API ${res.status} ${res.statusText} (${url})`);
    }
    const delay = Math.min(60_000, 1000 * Math.pow(2, attempt));
    process.stdout.write(`\n    [${res.status}] backing off ${delay}ms (attempt ${attempt + 1}/10)`);
    await sleep(delay);
  }
  throw new Error(`HF API failed after 10 retries (${url})`);
}

// Fetch the first n rows via the HF datasets-server. Caches each page to
// disk individually so a partial run isn't wasted — a re-run picks up exactly
// where the previous one stopped.
async function fetchRows(config, n, isCorpus) {
  const pageDir = path.join(CACHE_DIR, `${config}-pages`);
  fs.mkdirSync(pageDir, { recursive: true });
  const t0 = Date.now();
  for (let offset = 0; offset < n; offset += PAGE) {
    const length = Math.min(PAGE, n - offset);
    const pagePath = path.join(pageDir, `${String(offset).padStart(8, '0')}.json`);
    if (!fs.existsSync(pagePath)) {
      const page = await fetchPage(config, offset, length);
      fs.writeFileSync(pagePath, JSON.stringify(page.rows));
      // Throttle to ~5 req/s to stay below the rate limit.
      await sleep(200);
    }
    if (offset % (PAGE * 10) === 0 || offset + length >= n) {
      const rate = (offset + length) / Math.max(0.001, (Date.now() - t0) / 1000);
      process.stdout.write(`\r  fetching ${config}: ${Math.min(offset + length, n)}/${n} (${rate.toFixed(0)}/s)`);
    }
  }
  process.stdout.write('\n');
  // Concat pages in order.
  const texts = [];
  const files = fs.readdirSync(pageDir).filter((f) => f.endsWith('.json')).sort();
  for (const f of files) {
    const rows = JSON.parse(fs.readFileSync(path.join(pageDir, f), 'utf8'));
    for (const r of rows) {
      const row = r.row;
      let text = row.text ?? '';
      if (isCorpus && row.title) text = row.title + '. ' + text;
      if (text) texts.push(text);
      if (texts.length >= n) break;
    }
    if (texts.length >= n) break;
  }
  return texts.slice(0, n);
}

function writeFixture(outPath, vectors, n, d) {
  const buf = Buffer.alloc(16 + n * d * 4);
  MAGIC.copy(buf, 0);
  buf.writeUInt32LE(0, 4);
  buf.writeUInt32LE(d, 8);
  buf.writeUInt32LE(n, 12);
  Buffer.from(vectors.buffer, vectors.byteOffset, vectors.byteLength).copy(buf, 16);
  fs.writeFileSync(outPath, buf);
  console.log(`  wrote ${outPath} (${(buf.length / 1e6).toFixed(2)} MB)`);
}

async function main() {
  await fs.promises.mkdir(OUT_DIR, { recursive: true });
  await fs.promises.mkdir(CACHE_DIR, { recursive: true });

  const name = fixtureName(N);
  console.log(`--- build-msmarco-fixture (n=${N}, queries=${Q}, d=${D}, name=${name}) ---`);

  console.log(`1) fetching first ${N} passages and ${Q} queries from ${DATASET}`);
  const passages = await fetchRows('corpus', N, true);
  const queries = await fetchRows('queries', Q, false);
  console.log(`   passages: ${passages.length}, queries: ${queries.length}`);
  if (passages.length < N) throw new Error(`only got ${passages.length} passages, need ${N}`);
  if (queries.length < Q) throw new Error(`only got ${queries.length} queries, need ${Q}`);

  console.log(`2) loading ${MODEL} (first run downloads ~25 MB of weights)`);
  const { pipeline } = await import('@xenova/transformers');
  const embed = await pipeline('feature-extraction', MODEL, { quantized: true });

  const embedAll = async (texts, label) => {
    const out = new Float32Array(texts.length * D);
    let done = 0;
    const t0 = Date.now();
    for (let i = 0; i < texts.length; i += BATCH) {
      const batch = texts.slice(i, i + BATCH);
      const result = await embed(batch, { pooling: 'mean', normalize: true });
      out.set(result.data, i * D);
      done += batch.length;
      const rate = done / Math.max(0.001, (Date.now() - t0) / 1000);
      process.stdout.write(`\r   ${label}: ${done}/${texts.length} (${rate.toFixed(1)}/s)`);
    }
    process.stdout.write('\n');
    return out;
  };

  console.log(`3) embedding passages (batch=${BATCH})`);
  const corpus = await embedAll(passages, 'passages');
  console.log(`4) embedding queries`);
  const qv = await embedAll(queries, 'queries');

  console.log('5) writing fixtures');
  writeFixture(path.join(OUT_DIR, `${name}-corpus.bin`), corpus, N, D);
  writeFixture(path.join(OUT_DIR, `${name}-queries.bin`), qv, Q, D);
  console.log('done');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
