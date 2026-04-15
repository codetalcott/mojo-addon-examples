// @qkstat/rag — platform loader
//
// Each @qkstat/rag-<platform> sub-package ships a prebuilt rag.node (plus
// bundled runtime libs on Linux). This glue package resolves the right one
// and falls back to a local source build for development.

const path = require('path');

const PLATFORMS = {
  'darwin-arm64': '@qkstat/rag-darwin-arm64',
  'linux-x64': '@qkstat/rag-linux-x64',
};

const key = `${process.platform}-${process.arch}`;
const pkg = PLATFORMS[key];

const { GpuIndex } = require('./lib/GpuIndex.js');

let addon;
if (pkg) {
  try {
    addon = require(pkg);
  } catch {
    // Platform package not installed — fall back to local build (dev).
    try {
      addon = require(path.join(__dirname, 'build', 'rag.node'));
    } catch {
      throw new Error(
        `@qkstat/rag: no prebuilt for ${key}, and no local build at ` +
          `packages/rag/build/rag.node. Install from npm (\`npm i @qkstat/rag\`) ` +
          `or build from source: \`pixi run bash packages/rag/build.sh\` ` +
          `(requires Mojo nightly + MAX via pixi).`,
      );
    }
  }
} else {
  // Unsupported platform — attempt local build only.
  try {
    addon = require(path.join(__dirname, 'build', 'rag.node'));
  } catch {
    throw new Error(
      `@qkstat/rag: unsupported platform ${key}. ` +
        `Supported: ${Object.keys(PLATFORMS).join(', ')}.`,
    );
  }
}

module.exports = {
  loadMatrixGpu: addon.loadMatrixGpu,
  matmulHandle: addon.matmulHandle,
  searchHandle: addon.searchHandle,
  releaseMatrixGpu: addon.releaseMatrixGpu,
  GpuIndex,
};
