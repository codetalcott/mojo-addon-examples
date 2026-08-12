#!/usr/bin/env bash
#
# H100 warm-path bench for @qkstat/embed against real MS-MARCO passages.
#
# Builds packages/retrieve/build/retrieve.node and packages/embed/build/embed.node,
# regenerates the MS-MARCO JSONL fixtures (the embed bench needs raw text,
# not the .bin embeddings used by examples/matmul/matmul_rag.js), then runs
# packages/embed/bench.js across 6 shapes:
#   {1k, 10k} corpus × {batch-1, batch-8, batch-64} queries
# at the natural MS-MARCO sequence-length distribution (capped at 128 tokens
# by tokenize.js).
#
# Mirrors scripts/runpod-bench-3d.sh's shell-redirect convention. Output
# lands in $HOME/bench-embed-msmarco-h100.txt; commit it to
# docs/bench-embed-msmarco-h100.txt after scp-ing back.
#
# Usage on a fresh RunPod H100 shell (repo already cloned to ~/mojo-addon-examples):
#
#   cd ~/mojo-addon-examples && bash scripts/runpod-bench-embed.sh
#
# Or as a single curl (public repo):
#
#   curl -fsSL https://raw.githubusercontent.com/codetalcott/mojo-addon-examples/main/scripts/runpod-bench-embed.sh | bash
#
# Expected duration: ~10-15 min on a warm-volume pod (pixi install ~5 min
# first run, builds ~1 min, fixture build ~5 min, bench ~3 min). Cold-volume
# pods add ~5 min for HF model + dataset downloads. Terminate the pod
# immediately after cat-ing the output file.

set -e

OUTFILE="${HOME}/bench-embed-msmarco-h100.txt"

echo "=== EMBED MS-MARCO H100 WARM-PATH BENCH ==="  >  "$OUTFILE"
date                                                >> "$OUTFILE"
echo ""                                             >> "$OUTFILE"
nvidia-smi | head -20                               >> "$OUTFILE" 2>&1 || true
echo ""                                             >> "$OUTFILE"
uname -a                                            >> "$OUTFILE"

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
node --version                                      >> "$OUTFILE"
echo ""                                             >> "$OUTFILE"

# --- Locate or clone repo ------------------------------------------------
if [ -d "$HOME/mojo-addon-examples/.git" ]; then
    cd "$HOME/mojo-addon-examples"
elif [ -d "./.git" ] && [ -f "./packages/embed/bench.js" ]; then
    :  # Already in repo
else
    cd "$HOME"
    if [ ! -d mojo-addon-examples ]; then
        echo "ERROR: repo not found. Clone it first with:" >&2
        echo "  git clone https://github.com/codetalcott/mojo-addon-examples.git" >&2
        exit 1
    fi
    cd mojo-addon-examples
fi

git fetch origin main >/dev/null 2>&1 || true
git checkout main >/dev/null 2>&1 || true
git pull --ff-only >/dev/null 2>&1 || true
git rev-parse --short HEAD                          >> "$OUTFILE"
echo ""                                             >> "$OUTFILE"

# --- Install deps --------------------------------------------------------
pixi install
npm install >/dev/null

# --- Build addons (retrieve first; embed depends on retrieve at runtime) -----------
echo "=== BUILD: packages/retrieve/build/retrieve.node ==="   >> "$OUTFILE"
pixi run bash packages/retrieve/build.sh                 2>&1 | tail -3 >> "$OUTFILE"
echo ""                                             >> "$OUTFILE"
echo "=== BUILD: packages/embed/build/embed.node ===" >> "$OUTFILE"
pixi run bash packages/embed/build.sh               2>&1 | tail -3 >> "$OUTFILE"

# --- Ensure MS-MARCO JSONL fixtures exist --------------------------------
# The embed bench reads raw passage text from .jsonl (not .bin embeddings).
# build-msmarco-fixture.js writes both: .jsonl in step 1b, .bin in step 5.
# An older fixture run may have produced only .bin — re-run if .jsonl is
# missing.
CORPUS_JSONL="examples/rag-demo/fixtures/msmarco-10k-corpus.jsonl"
QUERIES_JSONL="examples/rag-demo/fixtures/msmarco-10k-queries.jsonl"
echo ""                                             >> "$OUTFILE"
if [ -f "$CORPUS_JSONL" ] && [ -f "$QUERIES_JSONL" ]; then
    echo "=== FIXTURES: msmarco-10k JSONL present, skipping build ==" >> "$OUTFILE"
else
    echo "=== BUILD: msmarco-10k fixture (JSONL + .bin) ==" >> "$OUTFILE"
    pixi run node scripts/build-msmarco-fixture.js  >> "$OUTFILE" 2>&1
fi

# --- Bench: tokenize + embed + search end-to-end ------------------------
echo ""                                             >> "$OUTFILE"
echo "=== BENCH: packages/embed/bench.js (msmarco-10k) ==" >> "$OUTFILE"
pixi run node packages/embed/bench.js               >> "$OUTFILE" 2>&1

echo ""                                             >> "$OUTFILE"
echo "=== nvidia-smi FINAL ==="                     >> "$OUTFILE"
nvidia-smi                                          >> "$OUTFILE" 2>&1

echo ""                                             >> "$OUTFILE"
echo "=== DONE ==="                                 >> "$OUTFILE"
date                                                >> "$OUTFILE"

echo ""
echo "DONE. Output in $OUTFILE ($(wc -l < "$OUTFILE") lines)"
echo "Run: cat $OUTFILE"
echo "Then scp back to docs/bench-embed-msmarco-h100.txt"
