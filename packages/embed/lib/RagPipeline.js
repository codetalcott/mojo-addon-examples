// RagPipeline — embed + index + search composed into one object.
//
// Owns an EmbeddingEngine and (after addTexts) a GpuIndex. Both addons load
// into the same Node process on separate CUDA contexts (the kernel-factory
// composition validated by the embedding-kernel spike).
//
// Designed so a long-lived service (MCP daemon, watch-mode reindexer) can:
//   - pay MAX cold-start at boot via warmup()
//   - reindex a corpus via addTexts(docs)
//   - serve queries via search(text, k) / searchTexts(texts, k)
// without manually plumbing engine.embed() output into new GpuIndex({...}).

const { EmbeddingEngine } = require('./EmbeddingEngine');
const { GpuIndex } = require('@qkstat/retrieve');

const DEFAULT_BATCH = 64;
const DEFAULT_DIM = 384;

class RagPipeline {
  constructor({ dim = DEFAULT_DIM, engine = null, batchSize = DEFAULT_BATCH } = {}) {
    this.dim = dim;
    this.batchSize = batchSize;
    this.engine = engine || new EmbeddingEngine();
    this.index = null;
  }

  get size() {
    return this.index ? this.index.n : 0;
  }

  async warmup() {
    await this.engine.warmup();
  }

  // Embed `docs` in batches of this.batchSize, build a fresh GpuIndex.
  // If a previous index exists, releases its GPU buffer first.
  async addTexts(docs) {
    if (!Array.isArray(docs) || docs.length === 0) {
      throw new Error('RagPipeline.addTexts: docs must be a non-empty string[]');
    }
    if (this.index) {
      this.index.close();
      this.index = null;
    }
    const t0 = performance.now();
    const embeddings = new Float32Array(docs.length * this.dim);
    for (let off = 0; off < docs.length; off += this.batchSize) {
      const slice = docs.slice(off, off + this.batchSize);
      const e = await this.engine.embedAsync(slice);
      embeddings.set(e, off * this.dim);
    }
    const embedMs = performance.now() - t0;
    this.index = new GpuIndex({ docs, embeddings, dim: this.dim });
    return { count: docs.length, embedMs };
  }

  async search(text, k = 10) {
    if (!this.index) {
      throw new Error('RagPipeline.search: no index — call addTexts(docs) first');
    }
    if (typeof text !== 'string' || text.length === 0) {
      throw new Error('RagPipeline.search: text must be a non-empty string');
    }
    const qEmb = await this.engine.embedAsync([text]);
    return this.index.searchAsync(qEmb, k);
  }

  // Returns an array of result lists, one per input query.
  async searchTexts(texts, k = 10) {
    if (!this.index) {
      throw new Error('RagPipeline.searchTexts: no index — call addTexts(docs) first');
    }
    if (!Array.isArray(texts) || texts.length === 0) {
      throw new Error('RagPipeline.searchTexts: texts must be a non-empty string[]');
    }
    const qEmb = await this.engine.embedAsync(texts);
    const { indices, scores, batch, k: kOut } = await this.index.searchBatchAsync(qEmb, k);
    const docs = this.index.docs;
    const out = new Array(batch);
    for (let b = 0; b < batch; b++) {
      const row = new Array(kOut);
      for (let i = 0; i < kOut; i++) {
        const idx = indices[b * kOut + i];
        row[i] = { doc: docs[idx], score: scores[b * kOut + i], index: idx };
      }
      out[b] = row;
    }
    return out;
  }

  close() {
    if (this.index) {
      this.index.close();
      this.index = null;
    }
  }
}

module.exports = { RagPipeline };
