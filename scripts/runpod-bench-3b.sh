#!/usr/bin/env bash
#
# Phase 3b.3 Track B: H100 validation runbook (self-contained).
#
# Fresh-RunPod bootstrap that installs tooling, clones the repo, builds all
# three cached addons (simd-search from 3a, image + stats from 3b), runs
# regression + benchmarks, and writes the combined output to
# ~/bench-cached-3b.txt.
#
# Usage on a fresh RunPod H100 shell:
#
#   curl -fsSL https://raw.githubusercontent.com/codetalcott/mojo-addon-examples/main/scripts/runpod-bench-3b.sh | bash
#
# Expected duration: ~10-15 minutes. Terminate the pod immediately after
# cat-ing the output file.

set -e

OUTFILE="${HOME}/bench-cached-3b.txt"

echo "=== PHASE 3B.3 H100 BENCH ==="         >  "$OUTFILE"
date                                        >> "$OUTFILE"
echo ""                                     >> "$OUTFILE"
nvidia-smi | head -20                       >> "$OUTFILE" 2>&1 || true
echo ""                                     >> "$OUTFILE"
uname -a                                    >> "$OUTFILE"

# --- Install pixi --------------------------------------------------------
if [ ! -x "$HOME/.pixi/bin/pixi" ]; then
    curl -fsSL https://pixi.sh/install.sh | bash
fi
export PATH="$HOME/.pixi/bin:$PATH"

# --- Install Node 22 via nvm ---------------------------------------------
if [ ! -s "$HOME/.nvm/nvm.sh" ]; then
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
fi
export NVM_DIR="$HOME/.nvm"
# shellcheck disable=SC1091
. "$NVM_DIR/nvm.sh"
nvm install 22
nvm use 22
node --version                              >> "$OUTFILE"
echo ""                                     >> "$OUTFILE"

# --- Clone repo ----------------------------------------------------------
cd "$HOME"
if [ ! -d mojo-addon-examples ]; then
    git clone https://github.com/codetalcott/mojo-addon-examples.git
fi
cd mojo-addon-examples
git fetch origin main
git checkout main
git pull --ff-only
git rev-parse --short HEAD                  >> "$OUTFILE"
echo ""                                     >> "$OUTFILE"

# --- Install deps --------------------------------------------------------
pixi install
npm install

# --- Build originals (oracles for cached regression tests) --------------
echo "=== BUILD: stats.node (oracle) ==="    >> "$OUTFILE"
pixi run bash stats/build.sh                2>&1 | tail -5 >> "$OUTFILE"

echo ""                                      >> "$OUTFILE"
echo "=== BUILD: image.node (oracle) ==="    >> "$OUTFILE"
pixi run bash image/build.sh                2>&1 | tail -5 >> "$OUTFILE"

echo ""                                      >> "$OUTFILE"
echo "=== BUILD: search.node (oracle) ==="   >> "$OUTFILE"
pixi run bash simd-search/build.sh          2>&1 | tail -5 >> "$OUTFILE"

# --- Build cached variants ----------------------------------------------
echo ""                                                   >> "$OUTFILE"
echo "=== BUILD: search_cached.node (3a) ==="             >> "$OUTFILE"
pixi run bash simd-search/build_cached.sh                2>&1 | tail -5 >> "$OUTFILE"

echo ""                                                   >> "$OUTFILE"
echo "=== BUILD: image_cached.node (3b.1) ==="            >> "$OUTFILE"
pixi run bash image/build_cached.sh                      2>&1 | tail -5 >> "$OUTFILE"

echo ""                                                   >> "$OUTFILE"
echo "=== BUILD: stats_cached.node (3b.2) ==="            >> "$OUTFILE"
pixi run bash stats/build_cached.sh                      2>&1 | tail -5 >> "$OUTFILE"

# --- Regression tests ----------------------------------------------------
echo ""                                                   >> "$OUTFILE"
echo "=== REGRESSION: search test_cached (3a) ==="        >> "$OUTFILE"
node --expose-gc simd-search/test_cached.js              >> "$OUTFILE" 2>&1

echo ""                                                   >> "$OUTFILE"
echo "=== REGRESSION: image test_cached (3b.1) ==="       >> "$OUTFILE"
node --expose-gc image/test_cached.js                    >> "$OUTFILE" 2>&1

echo ""                                                   >> "$OUTFILE"
echo "=== REGRESSION: stats test_cached (3b.2) ==="       >> "$OUTFILE"
node --expose-gc stats/test_cached.js                    >> "$OUTFILE" 2>&1

# --- Benchmarks ----------------------------------------------------------
echo ""                                                   >> "$OUTFILE"
echo "=== BENCH: search_cached (3a reference) ==="        >> "$OUTFILE"
node simd-search/search_cached.js                        >> "$OUTFILE" 2>&1

echo ""                                                   >> "$OUTFILE"
echo "=== BENCH: image_cached (3b.1) ==="                 >> "$OUTFILE"
node image/image_cached.js                               >> "$OUTFILE" 2>&1

echo ""                                                   >> "$OUTFILE"
echo "=== BENCH: stats_cached (3b.2) ==="                 >> "$OUTFILE"
node stats/stats_cached.js                               >> "$OUTFILE" 2>&1

echo ""                                                   >> "$OUTFILE"
echo "=== DONE ==="                                       >> "$OUTFILE"
date                                                     >> "$OUTFILE"

echo ""
echo "DONE. Output in $OUTFILE ($(wc -l < "$OUTFILE") lines)"
echo "Run: cat $OUTFILE"
