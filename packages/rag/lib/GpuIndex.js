// GpuIndex — high-level RAG wrapper around the four GPU primitives.
//
// Extracted from mojo-addon-examples/examples/rag-demo/search.js. Handles
// the row-major → column-major transpose so callers can pass embeddings
// in the natural [N, dim] layout.

const path = require('path');

function resolveAddon() {
  // Require with a lazy resolve so unit tests can inject their own addon.
  const pkg = process.platform + '-' + process.arch;
  const map = {
    'darwin-arm64': '@qkstat/rag-darwin-arm64',
    'linux-x64': '@qkstat/rag-linux-x64',
  };
  const platformPkg = map[pkg];
  if (platformPkg) {
    try { return require(platformPkg); } catch { /* fall through */ }
  }
  return require(path.join(__dirname, '..', 'build', 'rag.node'));
}

class GpuIndex {
  constructor({ docs, embeddings, dim, addon }) {
    this.addon = addon || resolveAddon();
    this.docs = docs;
    this.dim = dim;
    this.n = docs.length;

    // Row-major [N, dim] → column-major [dim, N] so query @ corpus = [1, N].
    const corpusT = new Float32Array(dim * this.n);
    for (let i = 0; i < this.n; i++) {
      for (let j = 0; j < dim; j++) {
        corpusT[j * this.n + i] = embeddings[i * dim + j];
      }
    }
    this.hCorpus = this.addon.loadMatrixGpu(corpusT, dim, this.n);
  }

  search(queryEmbedding, k) {
    const hQuery = this.addon.loadMatrixGpu(queryEmbedding, 1, this.dim);
    const idx = new Uint32Array(k);
    const scores = new Float32Array(k);
    this.addon.searchHandle(hQuery, this.hCorpus, idx, scores);
    this.addon.releaseMatrixGpu(hQuery);
    const results = new Array(k);
    for (let i = 0; i < k; i++) {
      results[i] = { doc: this.docs[idx[i]], score: scores[i], index: idx[i] };
    }
    return results;
  }

  close() {
    this.addon.releaseMatrixGpu(this.hCorpus);
  }
}

module.exports = { GpuIndex };
