#!/usr/bin/env bash
#
# Phase 3c.3 Track B: H100 validation runbook for cached matmul (self-contained).
#
# Builds matmul_cached.node (new in Phase 3c.2, uses MAX's linalg.matmul),
# runs regression tests, and runs the 3-path benchmark at 256/512/1024/2048.
# Writes combined output to ~/bench-cached-3c.txt.
#
# Usage on a fresh RunPod H100 shell (repo already cloned, e.g. via
# `git clone https://${GH_TOKEN}@github.com/codetalcott/mojo-addon-examples.git`):
#
#   cd ~/mojo-addon-examples && bash scripts/runpod-bench-3c.sh
#
# Or as a single curl (if the repo is public / token is set in env):
#
#   curl -fsSL https://raw.githubusercontent.com/codetalcott/mojo-addon-examples/main/scripts/runpod-bench-3c.sh | bash
#
# Expected duration: ~5-10 minutes. Terminate the pod immediately after
# cat-ing the output file.

set -e

OUTFILE="${HOME}/bench-cached-3c.txt"

echo "=== PHASE 3C.3 H100 MATMUL BENCH ==="  >  "$OUTFILE"
date                                         >> "$OUTFILE"
echo ""                                      >> "$OUTFILE"
nvidia-smi | head -20                        >> "$OUTFILE" 2>&1 || true
echo ""                                      >> "$OUTFILE"
uname -a                                     >> "$OUTFILE"

# --- Install pixi (skip if already present) ------------------------------
if [ ! -x "$HOME/.pixi/bin/pixi" ]; then
    curl -fsSL https://pixi.sh/install.sh | bash
fi
export PATH="$HOME/.pixi/bin:$PATH"

# --- Install Node 22 via nvm (skip if already present) -------------------
if [ ! -s "$HOME/.nvm/nvm.sh" ]; then
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
fi
export NVM_DIR="$HOME/.nvm"
# shellcheck disable=SC1091
. "$NVM_DIR/nvm.sh"
nvm install 22 >/dev/null
nvm use 22 >/dev/null
node --version                               >> "$OUTFILE"
echo ""                                      >> "$OUTFILE"

# --- Locate or clone repo ------------------------------------------------
if [ -d "$HOME/mojo-addon-examples/.git" ]; then
    cd "$HOME/mojo-addon-examples"
elif [ -d "./.git" ] && [ -f "./examples/matmul/addon_cached.mojo" ]; then
    :  # Already in repo
else
    cd "$HOME"
    if [ ! -d mojo-addon-examples ]; then
        echo "ERROR: repo not found. Clone it first with:" >&2
        echo "  git clone https://\${GH_TOKEN}@github.com/codetalcott/mojo-addon-examples.git" >&2
        exit 1
    fi
    cd mojo-addon-examples
fi

git fetch origin main >/dev/null 2>&1 || true
git checkout main >/dev/null 2>&1 || true
git pull --ff-only >/dev/null 2>&1 || true
git rev-parse --short HEAD                   >> "$OUTFILE"
echo ""                                      >> "$OUTFILE"

# --- Install deps (pixi picks up the max package from pixi.toml) --------
# This is the slow step on a fresh pod (~5 min to download MAX nightly).
# Subsequent runs are cached.
pixi install
npm install >/dev/null

# --- Build CPU-only matmul.node (oracle for CPU row) ---------------------
echo "=== BUILD: matmul.node (oracle) ==="   >> "$OUTFILE"
pixi run bash examples/matmul/build.sh                2>&1 | tail -3 >> "$OUTFILE"

# --- Build cached matmul with tensor cores ------------------------------
echo ""                                      >> "$OUTFILE"
echo "=== BUILD: matmul_cached.node (3c.2) ==="  >> "$OUTFILE"
pixi run bash examples/matmul/build_cached.sh         2>&1 | tail -3 >> "$OUTFILE"

# --- Regression test ----------------------------------------------------
echo ""                                      >> "$OUTFILE"
echo "=== REGRESSION: matmul test_cached (3c.2) ==="  >> "$OUTFILE"
node examples/matmul/test_cached.js                   >> "$OUTFILE" 2>&1

# --- 3-path benchmark ---------------------------------------------------
echo ""                                      >> "$OUTFILE"
echo "=== BENCH: matmul_cached (3c.2) ==="   >> "$OUTFILE"
node examples/matmul/matmul_cached.js                 >> "$OUTFILE" 2>&1

echo ""                                      >> "$OUTFILE"
echo "=== DONE ==="                          >> "$OUTFILE"
date                                         >> "$OUTFILE"

echo ""
echo "DONE. Output in $OUTFILE ($(wc -l < "$OUTFILE") lines)"
echo "Run: cat $OUTFILE"
