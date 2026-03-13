// demo/server.js — Express server for mojo-addon-examples live demo
//
// Serves a single-page showcase and 5 API endpoints that run the Mojo addons.
// The addons are pre-built .node files — no Mojo runtime needed at serving time.

const express = require('express');
const path = require('path');
const { performance } = require('perf_hooks');

const app = express();
const PORT = process.env.PORT || 8080;

// --- Load addons -----------------------------------------------------------

const imageAddon = require('../image/build/image.node');
const statsAddon = require('../stats/build/stats.node');
const searchAddon = require('../simd-search/build/search.node');
const matmulAddon = require('../matmul/build/matmul.node');
const hashAddon = require('../wyhash/build/wyhash.node');

// --- Static files -----------------------------------------------------------

app.use(express.static(path.join(__dirname, 'public')));

// --- JS baselines (run server-side for fair comparison) --------------------

function jsGrayscale(rgba, w, h) {
  const out = new Uint8Array(rgba.length);
  for (let i = 0; i < w * h; i++) {
    const off = i * 4;
    const gray = (77 * rgba[off] + 150 * rgba[off+1] + 29 * rgba[off+2]) >> 8;
    out[off] = out[off+1] = out[off+2] = gray;
    out[off+3] = rgba[off+3];
  }
  return out;
}

function jsStats(data) {
  let sum = 0, min = data[0], max = data[0];
  for (let i = 0; i < data.length; i++) {
    sum += data[i];
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }
  const mean = sum / data.length;
  let sumSq = 0;
  for (let i = 0; i < data.length; i++) {
    const d = data[i] - mean;
    sumSq += d * d;
  }
  const stddev = Math.sqrt(sumSq / data.length);
  // Sort for percentiles (matches what Mojo computes)
  const sorted = Float64Array.from(data).sort();
  const p50 = sorted[Math.floor((data.length - 1) * 0.5)];
  const p95 = sorted[Math.floor((data.length - 1) * 0.95)];
  const p99 = sorted[Math.floor((data.length - 1) * 0.99)];
  return { mean, stddev, min, max, p50, p95, p99 };
}

function jsCountLines(buf) {
  let count = 0;
  for (let i = 0; i < buf.length; i++) {
    if (buf[i] === 0x0A) count++;
  }
  return count;
}

function jsMatmulNaive(a, b, c, M, K, N) {
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let p = 0; p < K; p++) {
        sum += a[i * K + p] * b[p * N + j];
      }
      c[i * N + j] = sum;
    }
  }
}

function jsFnv1a(buf) {
  let hash = 2166136261;
  for (let i = 0; i < buf.length; i++) {
    hash ^= buf[i];
    hash = (hash * 16777619) | 0;
  }
  return hash >>> 0;
}

// JS wyhash (BigInt) — same algorithm as the Mojo addon for fair comparison
const _WYP0 = 0xa0761d6478bd642fn;
const _WYP1 = 0xe7037ed1a0b428dbn;
const _WYP2 = 0x8ebc6af09c88c6e3n;
const _WYP3 = 0x589965cc75374cc3n;
const MASK64 = (1n << 64n) - 1n;

function _wymum(a, b) {
  const full = a * b;
  return (full & MASK64) ^ ((full >> 64n) & MASK64);
}

function jsWyhash(buf, seed = 0n) {
  seed = BigInt.asUintN(64, seed);
  const len = buf.length;
  seed ^= _wymum(seed ^ _WYP0, _WYP1);
  seed = BigInt.asUintN(64, seed);
  let a = 0n, b = 0n;
  if (len <= 16) {
    if (len >= 4) {
      a = (BigInt(buf.readUInt32LE(0)) << 32n) | BigInt(buf.readUInt32LE((len >> 3) << 2));
      b = (BigInt(buf.readUInt32LE(len - 4)) << 32n) | BigInt(buf.readUInt32LE(len - ((len >> 3) << 2) - 4));
    } else if (len > 0) {
      a = (BigInt(buf[0]) << 16n) | (BigInt(buf[len >> 1]) << 8n) | BigInt(buf[len - 1]);
    }
  } else if (len <= 48) {
    seed = BigInt.asUintN(64, _wymum(buf.readBigUInt64LE(0) ^ _WYP1, buf.readBigUInt64LE(8) ^ seed));
    if (len > 32) seed = BigInt.asUintN(64, _wymum(buf.readBigUInt64LE(16) ^ _WYP2, buf.readBigUInt64LE(24) ^ seed));
    a = buf.readBigUInt64LE(len - 16);
    b = buf.readBigUInt64LE(len - 8);
  } else {
    let see1 = seed, see2 = seed, i = 0, remaining = len;
    while (remaining > 48) {
      seed = BigInt.asUintN(64, _wymum(buf.readBigUInt64LE(i) ^ _WYP1, buf.readBigUInt64LE(i + 8) ^ seed));
      see1 = BigInt.asUintN(64, _wymum(buf.readBigUInt64LE(i + 16) ^ _WYP2, buf.readBigUInt64LE(i + 24) ^ see1));
      see2 = BigInt.asUintN(64, _wymum(buf.readBigUInt64LE(i + 32) ^ _WYP3, buf.readBigUInt64LE(i + 40) ^ see2));
      i += 48; remaining -= 48;
    }
    seed ^= see1 ^ see2; seed = BigInt.asUintN(64, seed);
    const tail = len - remaining;
    if (remaining > 32) {
      seed = BigInt.asUintN(64, _wymum(buf.readBigUInt64LE(tail) ^ _WYP1, buf.readBigUInt64LE(tail + 8) ^ seed));
      see1 = BigInt.asUintN(64, _wymum(buf.readBigUInt64LE(tail + 16) ^ _WYP2, buf.readBigUInt64LE(tail + 24) ^ see1));
      seed ^= see1; seed = BigInt.asUintN(64, seed);
    } else if (remaining > 16) {
      seed = BigInt.asUintN(64, _wymum(buf.readBigUInt64LE(tail) ^ _WYP1, buf.readBigUInt64LE(tail + 8) ^ seed));
    }
    a = buf.readBigUInt64LE(len - 16);
    b = buf.readBigUInt64LE(len - 8);
  }
  return BigInt.asUintN(64, _wymum(BigInt.asUintN(64, _WYP1 ^ BigInt(len)), _wymum(BigInt.asUintN(64, a ^ _WYP1), BigInt.asUintN(64, b ^ seed))));
}

// --- Benchmarking utility ---------------------------------------------------

function bench(fn, warmup = 3, iters = 10) {
  for (let i = 0; i < warmup; i++) fn();
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) fn();
  return (performance.now() - t0) / iters;
}

// --- API: Image processing --------------------------------------------------

app.get('/api/image/demo', async (req, res) => {
  try {
    let sharp;
    try {
      sharp = require('sharp');
    } catch {
      // sharp not available — use synthetic image
      return sendSyntheticImageDemo(res);
    }

    // Generate a colorful test image (gradient)
    const width = 640;
    const height = 480;
    const rgba = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        rgba[i]     = (x * 255 / (width - 1)) | 0;
        rgba[i + 1] = (y * 255 / (height - 1)) | 0;
        rgba[i + 2] = Math.min(255, Math.sqrt((x - width/2)**2 + (y - height/2)**2) * 255 / (width/2)) | 0;
        rgba[i + 3] = 255;
      }
    }

    // Run Mojo transforms with bench() for consistent timing methodology
    const transforms = {};

    transforms.grayscale = { ms: bench(() => imageAddon.grayscale(rgba, width, height), 3, 10) };
    transforms.brightness = { ms: bench(() => imageAddon.brightness(rgba, width, height, 1.5), 3, 10) };
    transforms.threshold = { ms: bench(() => imageAddon.threshold(rgba, width, height, 128), 3, 10) };
    transforms.blur = { ms: bench(() => imageAddon.blur(rgba, width, height, 5), 3, 10) };

    // Get actual result buffers for PNG encoding (post-bench)
    const grayRgba = imageAddon.grayscale(rgba, width, height);
    const brightRgba = imageAddon.brightness(rgba, width, height, 1.5);
    const threshRgba = imageAddon.threshold(rgba, width, height, 128);
    const blurRgba = imageAddon.blur(rgba, width, height, 5);

    // JS baseline for grayscale (same bench methodology)
    const jsMs = bench(() => jsGrayscale(rgba, width, height), 3, 10);
    transforms.jsGrayscaleMs = jsMs;

    // Convert to base64 PNG via sharp
    const toPng = async (data) => {
      const buf = await sharp(Buffer.from(data.buffer), { raw: { width, height, channels: 4 } })
        .png({ compressionLevel: 6 })
        .toBuffer();
      return buf.toString('base64');
    };

    const [origPng, grayPng, brightPng, threshPng, blurPng] = await Promise.all([
      toPng(rgba), toPng(grayRgba), toPng(brightRgba), toPng(threshRgba), toPng(blurRgba),
    ]);

    res.json({
      width, height,
      transforms,
      images: {
        original: origPng,
        grayscale: grayPng,
        brightness: brightPng,
        threshold: threshPng,
        blur: blurPng,
      },
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

function sendSyntheticImageDemo(res) {
  // Fallback: run transforms with timing but no PNG output
  const width = 1280, height = 720;
  const rgba = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      rgba[i] = (x * 255 / (width - 1)) | 0;
      rgba[i + 1] = (y * 255 / (height - 1)) | 0;
      rgba[i + 2] = 128;
      rgba[i + 3] = 255;
    }
  }

  const transforms = {};

  transforms.grayscale = { ms: bench(() => imageAddon.grayscale(rgba, width, height), 3, 10) };
  transforms.brightness = { ms: bench(() => imageAddon.brightness(rgba, width, height, 1.5), 3, 10) };
  transforms.threshold = { ms: bench(() => imageAddon.threshold(rgba, width, height, 128), 3, 10) };
  transforms.blur = { ms: bench(() => imageAddon.blur(rgba, width, height, 5), 3, 10) };
  transforms.jsGrayscaleMs = bench(() => jsGrayscale(rgba, width, height), 3, 10);

  res.json({ width, height, transforms, images: null });
}

// --- API: Stats -------------------------------------------------------------

app.get('/api/stats/demo', (req, res) => {
  try {
    const SIZE = 1_000_000;
    const data = new Float64Array(SIZE);
    for (let i = 0; i < SIZE; i++) data[i] = Math.random() * 1000;

    // Mojo stats (with warmup for fair timing)
    const mojoStatsMs = bench(() => statsAddon.stats(data), 3, 10);
    const result = statsAddon.stats(data);

    // Mojo histogram
    const mojoHistMs = bench(() => statsAddon.histogram(data, 50), 3, 10);
    const hist = statsAddon.histogram(data, 50);

    // JS baseline (includes sort for percentiles, same as Mojo)
    const jsMs = bench(() => jsStats(data), 2, 5);

    res.json({
      size: SIZE,
      stats: result,
      histogram: Array.from(hist),
      mojoStatsMs,
      mojoHistMs,
      jsMs,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// --- API: SIMD Search -------------------------------------------------------

// Pre-generate a text buffer (simulated log file)
const SEARCH_BUF_SIZE = 16 * 1024 * 1024; // 16MB
const searchBuf = Buffer.alloc(SEARCH_BUF_SIZE);
for (let i = 0; i < SEARCH_BUF_SIZE; i++) {
  searchBuf[i] = Math.random() < 0.01 ? 0x0A : Math.floor(Math.random() * 94) + 32;
}
// Inject some "ERROR" patterns
const errorBuf = Buffer.from('ERROR');
for (let i = 0; i < SEARCH_BUF_SIZE / 1000; i++) {
  const pos = Math.floor(Math.random() * (SEARCH_BUF_SIZE - 5));
  errorBuf.copy(searchBuf, pos);
}

app.get('/api/search/demo', (req, res) => {
  try {
    // Mojo countLines (warmed, multi-iteration)
    const mojoLinesMs = bench(() => searchAddon.countLines(searchBuf), 3, 10);
    const lineCount = searchAddon.countLines(searchBuf);

    // Mojo searchAll (warmed, multi-iteration)
    const mojoSearchMs = bench(() => searchAddon.searchAll(searchBuf, errorBuf), 3, 10);
    const positions = searchAddon.searchAll(searchBuf, errorBuf);

    // JS baselines (same bench methodology)
    const jsLinesMs = bench(() => jsCountLines(searchBuf), 3, 10);

    res.json({
      bufferSize: SEARCH_BUF_SIZE,
      lineCount,
      matchCount: positions.length,
      pattern: 'ERROR',
      mojoLinesMs,
      mojoSearchMs,
      jsLinesMs,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// --- API: Matrix Multiply ---------------------------------------------------

app.get('/api/matmul/demo', (req, res) => {
  try {
    const SIZE = 512;
    const a = new Float64Array(SIZE * SIZE);
    const b = new Float64Array(SIZE * SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      a[i] = Math.random();
      b[i] = Math.random();
    }

    const results = {};

    // JS baseline (naive) — same warmup/iter pattern for fair timing
    const cJs = new Float64Array(SIZE * SIZE);
    results.jsNaiveMs = bench(() => jsMatmulNaive(a, b, cJs, SIZE, SIZE, SIZE), 1, 3);

    // Mojo naive
    const c1 = new Float64Array(SIZE * SIZE);
    results.naiveMs = bench(() => matmulAddon.matmulNaive(a, b, c1, SIZE, SIZE, SIZE), 1, 3);

    // Mojo vectorized
    const c2 = new Float64Array(SIZE * SIZE);
    results.vectorizedMs = bench(() => matmulAddon.matmulVectorized(a, b, c2, SIZE, SIZE, SIZE), 1, 5);

    // Mojo tiled
    const c3 = new Float64Array(SIZE * SIZE);
    results.tiledMs = bench(() => matmulAddon.matmulTiled(a, b, c3, SIZE, SIZE, SIZE), 1, 5);

    // Mojo parallel
    const c4 = new Float64Array(SIZE * SIZE);
    results.parallelMs = bench(() => matmulAddon.matmulParallel(a, b, c4, SIZE, SIZE, SIZE), 2, 10);

    results.size = SIZE;

    res.json(results);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// --- API: Wyhash ------------------------------------------------------------

app.get('/api/hash/demo', (req, res) => {
  try {
    const sizes = [1024, 65536, 1048576, 16777216];
    const results = [];

    for (const size of sizes) {
      const buf = Buffer.alloc(size);
      for (let i = 0; i < size; i++) buf[i] = (i * 7 + 13) & 0xFF;

      const iters = size <= 65536 ? 1000 : size <= 1048576 ? 100 : 10;
      const mojoMs = bench(() => hashAddon.wyHash64(buf), 3, iters);
      // Same algorithm (wyhash) in JS BigInt — apples-to-apples comparison
      const jsWyhashMs = bench(() => jsWyhash(buf), 3, Math.max(1, Math.floor(iters / 10)));
      // Also show FNV-1a as a "best JS can do with Number arithmetic" reference
      const jsFnvMs = bench(() => jsFnv1a(buf), 3, iters);

      const mojoGBs = (size / 1e9) / (mojoMs / 1000);
      const jsWyhashGBs = (size / 1e9) / (jsWyhashMs / 1000);
      const jsFnvGBs = (size / 1e9) / (jsFnvMs / 1000);

      results.push({
        size,
        sizeLabel: size >= 1048576 ? `${size / 1048576}MB` : `${size / 1024}KB`,
        mojoMs: +mojoMs.toFixed(4),
        jsWyhashMs: +jsWyhashMs.toFixed(4),
        jsFnvMs: +jsFnvMs.toFixed(4),
        mojoGBs: +mojoGBs.toFixed(2),
        jsWyhashGBs: +jsWyhashGBs.toFixed(2),
        jsFnvGBs: +jsFnvGBs.toFixed(2),
        // Primary speedup: same algorithm comparison (wyhash vs wyhash)
        speedup: +(jsWyhashMs / mojoMs).toFixed(1),
      });
    }

    res.json({ results });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// --- Start ------------------------------------------------------------------

app.listen(PORT, () => {
  console.log(`Mojo addon demo running at http://localhost:${PORT}`);
});
