// EmbeddingEngine — high-level wrapper around the embedTokens primitive.
//
// Tokenizes via @huggingface/transformers (WordPiece for MiniLM-L6-v2),
// allocates the output Float32Array, and calls the Mojo addon. The first
// call pays MAX graph compile cost (~30s on H100 cold, much less warm).

const path = require('path');
const { tokenize } = require('../tokenize');

const EMBED_DIM = 384;

function resolveAddon() {
  const pkg = process.platform + '-' + process.arch;
  const map = {
    'darwin-arm64': '@qkstat/embed-darwin-arm64',
    'linux-x64': '@qkstat/embed-linux-x64',
  };
  const platformPkg = map[pkg];
  if (platformPkg) {
    try { return require(platformPkg); } catch { /* fall through */ }
  }
  return require(path.join(__dirname, '..', 'build', 'embed.node'));
}

class EmbeddingEngine {
  constructor({ addon } = {}) {
    this.addon = addon || resolveAddon();
    if (typeof this.addon.embedTokens !== 'function') {
      throw new Error('EmbeddingEngine: addon missing embedTokens — rebuild packages/embed');
    }
    this.dim = EMBED_DIM;
    this._warmed = false;
  }

  // Pay the MAX cold-start explicitly so a long-lived service can do it at
  // boot rather than on the first user query. Idempotent.
  async warmup() {
    if (this._warmed) return;
    await this.embedAsync(['.']);
    this._warmed = true;
  }

  async embed(texts) {
    if (!Array.isArray(texts) || texts.length === 0) {
      throw new Error('EmbeddingEngine.embed: texts must be a non-empty string[]');
    }
    const { ids, mask, batch, seqLen } = await tokenize(texts);
    const dst = new Float32Array(batch * EMBED_DIM);
    this.addon.embedTokens(ids, mask, batch, seqLen, dst);
    return dst;
  }

  async embedAsync(texts) {
    if (!Array.isArray(texts) || texts.length === 0) {
      throw new Error('EmbeddingEngine.embedAsync: texts must be a non-empty string[]');
    }
    if (typeof this.addon.embedTokensAsync !== 'function') {
      throw new Error('EmbeddingEngine.embedAsync: addon missing embedTokensAsync — rebuild packages/embed');
    }
    const { ids, mask, batch, seqLen } = await tokenize(texts);
    const dst = new Float32Array(batch * EMBED_DIM);
    await this.addon.embedTokensAsync(ids, mask, batch, seqLen, dst);
    return dst;
  }
}

module.exports = { EmbeddingEngine };
