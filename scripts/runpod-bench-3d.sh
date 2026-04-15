#!/usr/bin/env bash
#
# Phase 3d H100 validation: RAG-shape matmul + searchHandle + ANN baselines.
#
# Builds matmul_cached.node, runs test_cached.js regression, and executes the
# three-path RAG bench: JS baseline, onnxruntime-node CPU MatMul, cached GPU
# (matmulHandle + searchHandle), and hnswlib-node approximate ANN. The --full
# flag enables 1M-corpus shapes that crash on M4 Metal due to unified-memory
# limits but fit comfortably in H100 80GB HBM3.
#
# Usage on a fresh RunPod H100 shell (repo already cloned):
#
#   cd ~/mojo-addon-examples && bash scripts/runpod-bench-3d.sh
#
# Add FIXTURE=1 to also build + run the real-embedding (MS-MARCO + MiniLM)
# bench. Pre-staged .bin files under examples/rag-demo/fixtures/ auto-enable
# it without building:
#
#   FIXTURE=1 bash scripts/runpod-bench-3d.sh
#
# Or as a single curl (public repo):
#
#   curl -fsSL https://raw.githubusercontent.com/codetalcott/mojo-addon-examples/main/scripts/runpod-bench-3d.sh | bash
#
# Expected duration: ~10-20 minutes (pixi install ~5 min first run,
# hnswlib build ~2-5 min, bench runs total ~3-5 min). Terminate the pod
# immediately after cat-ing the output file.

set -e

OUTFILE="${HOME}/bench-rag-3d.txt"

echo "=== PHASE 3D H100 RAG-SHAPE BENCH ==="  >  "$OUTFILE"
date                                          >> "$OUTFILE"
echo ""                                       >> "$OUTFILE"
nvidia-smi | head -20                         >> "$OUTFILE" 2>&1 || true
echo ""                                       >> "$OUTFILE"
uname -a                                      >> "$OUTFILE"

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
node --version                                >> "$OUTFILE"
echo ""                                       >> "$OUTFILE"

# --- Locate or clone repo ------------------------------------------------
if [ -d "$HOME/mojo-addon-examples/.git" ]; then
    cd "$HOME/mojo-addon-examples"
elif [ -d "./.git" ] && [ -f "./matmul/addon_cached.mojo" ]; then
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
git rev-parse --short HEAD                    >> "$OUTFILE"
echo ""                                       >> "$OUTFILE"

# --- Install deps --------------------------------------------------------
# npm install pulls in napi-mojo, onnxruntime-node, hnswlib-node (all with
# x86_64 Linux prebuilds — no build-essential needed).
pixi install
npm install >/dev/null

# --- Build cached matmul (tensor cores via linalg.matmul) ----------------
echo "=== BUILD: matmul_cached.node ==="      >> "$OUTFILE"
pixi run bash matmul/build_cached.sh          2>&1 | tail -3 >> "$OUTFILE"

# --- Regression: test_cached.js (includes searchHandle cases) ------------
echo ""                                       >> "$OUTFILE"
echo "=== REGRESSION: matmul test_cached ===" >> "$OUTFILE"
node matmul/test_cached.js                    >> "$OUTFILE" 2>&1

# --- RAG-shape bench: 100k + 1M corpora, concurrency=100, all baselines --
echo ""                                       >> "$OUTFILE"
echo "=== BENCH: matmul_rag (3d, --full) ===" >> "$OUTFILE"
node matmul/matmul_rag.js --full --concurrency=100  >> "$OUTFILE" 2>&1

# --- Real-embedding fixture bench ----------------------------------------
# Skipped by default: fetches ~10k MS-MARCO rows from HF (rate-limited, ~5 min)
# and CPU-embeds them with Xenova/all-MiniLM-L6-v2 (~5 min) before running the
# bench at [1, 384] × [384, 10k]. Enable with FIXTURE=1 in the environment, or
# pre-stage the .bin files under examples/rag-demo/fixtures/ (e.g. scp from
# the machine where you ran build-msmarco-fixture.js) — if both files are
# already present the script skips the build and just runs the bench.
CORPUS_BIN="examples/rag-demo/fixtures/msmarco-10k-corpus.bin"
QUERIES_BIN="examples/rag-demo/fixtures/msmarco-10k-queries.bin"
if [ -f "$CORPUS_BIN" ] && [ -f "$QUERIES_BIN" ]; then
    RUN_FIXTURE=1
elif [ "${FIXTURE:-0}" = "1" ]; then
    RUN_FIXTURE=1
    echo ""                                                         >> "$OUTFILE"
    echo "=== BUILD: msmarco-10k fixture ==="                        >> "$OUTFILE"
    node scripts/build-msmarco-fixture.js                            >> "$OUTFILE" 2>&1
else
    RUN_FIXTURE=0
    echo ""                                                         >> "$OUTFILE"
    echo "=== SKIP: msmarco-10k fixture (set FIXTURE=1 to build) ==" >> "$OUTFILE"
fi

if [ "$RUN_FIXTURE" = "1" ]; then
    echo ""                                                         >> "$OUTFILE"
    echo "=== BENCH: matmul_rag (msmarco-10k real embeddings) ==="  >> "$OUTFILE"
    node matmul/matmul_rag.js --fixture=msmarco-10k --concurrency=100  >> "$OUTFILE" 2>&1
fi

echo ""                                       >> "$OUTFILE"
echo "=== nvidia-smi FINAL ==="               >> "$OUTFILE"
nvidia-smi                                    >> "$OUTFILE" 2>&1

echo ""                                       >> "$OUTFILE"
echo "=== DONE ==="                           >> "$OUTFILE"
date                                          >> "$OUTFILE"

echo ""
echo "DONE. Output in $OUTFILE ($(wc -l < "$OUTFILE") lines)"
echo "Run: cat $OUTFILE"
