// @qkstat/embed — platform loader
//
// Each @qkstat/embed-<platform> sub-package ships a prebuilt embed.node
// (plus bundled MAX runtime libs on Linux). This glue package resolves the
// right one and falls back to a local source build for development.

const path = require('path');

const PLATFORMS = {
  'darwin-arm64': '@qkstat/embed-darwin-arm64',
  'linux-x64': '@qkstat/embed-linux-x64',
};

const key = `${process.platform}-${process.arch}`;
const pkg = PLATFORMS[key];

const { EmbeddingEngine } = require('./lib/EmbeddingEngine.js');
const { RagPipeline } = require('./lib/RagPipeline.js');

let addon;
if (pkg) {
  try {
    addon = require(pkg);
  } catch {
    // Platform package not installed — fall back to local build (dev).
    try {
      addon = require(path.join(__dirname, 'build', 'embed.node'));
    } catch {
      throw new Error(
        `@qkstat/embed: no prebuilt for ${key}, and no local build at ` +
          `packages/embed/build/embed.node. Install from npm (\`npm i @qkstat/embed\`) ` +
          `or build from source: \`pixi run bash packages/embed/build.sh\` ` +
          `(requires Mojo nightly + MAX via pixi).`,
      );
    }
  }
} else {
  // Unsupported platform — attempt local build only.
  try {
    addon = require(path.join(__dirname, 'build', 'embed.node'));
  } catch {
    throw new Error(
      `@qkstat/embed: unsupported platform ${key}. ` +
        `Supported: ${Object.keys(PLATFORMS).join(', ')}.`,
    );
  }
}

module.exports = {
  embedTokens: addon.embedTokens,
  embedTokensAsync: addon.embedTokensAsync,
  EmbeddingEngine,
  RagPipeline,
};
