#!/usr/bin/env bash
# scripts/verify-all.sh — build and test every target on the local machine.
#
# The comprehensive gate. CI cannot be this thorough: hosted runners have no
# NVIDIA GPU (ubuntu) and unproven Metal-in-a-VM (macos), so every GPU-execution
# test in this repo — the four *_cached suites and packages/retrieve's Jest — can
# only run here or on a pod. CI is therefore compile-only; this script is where
# GPU correctness is actually established before a push.
#
# Companion scripts:
#   scripts/verify-gpu-h100.sh — same survey, but for a RunPod H100 (sm_90,
#                                syncs a git ref, checks nvidia-smi).
#   This one is platform-agnostic and never touches git.
#
# Deliberately does NOT use `set -e`. This is a survey: one failing target must
# not hide the others. `npm test` in particular is &&-chained, so it stops at
# the first bad addon — this script runs each test file on its own so a single
# stale .node cannot mask the rest.
#
# Usage (from anywhere):
#   bash scripts/verify-all.sh                  # everything except embed's runtime test
#   bash scripts/verify-all.sh --with-embed-test  # ...including it (needs model weights)
#   bash scripts/verify-all.sh --skip-embed       # skip embed entirely (fastest)

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

WITH_EMBED_TEST=0
SKIP_EMBED=0
for arg in "$@"; do
  case "$arg" in
    --with-embed-test) WITH_EMBED_TEST=1 ;;
    --skip-embed)      SKIP_EMBED=1 ;;
    -h|--help)         sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown flag: $arg (try --help)" >&2; exit 2 ;;
  esac
done

FAILED=0
declare -a RESULTS=()
# NOT `mktemp -t verify-all`: that is a BSD-ism. GNU coreutils treats -t's
# argument as a template and rejects one without a trailing XXX run, so the
# BSD spelling dies with "too few X's in template" on Linux — which is every
# pod and CI runner. An explicit path with XXXXXX works on both.
LOG="$(mktemp "${TMPDIR:-/tmp}/verify-all.XXXXXX")"
trap 'rm -f "$LOG"' EXIT

step() {  # step <label> <cmd...>
  local label="$1"; shift
  printf '::: %s\n' "$label"
  if "$@" > "$LOG" 2>&1; then
    RESULTS+=("PASS  $label")
    echo "    PASS"
  else
    RESULTS+=("FAIL  $label")
    FAILED=1
    # Short logs (a lone ABORT from a stale .node) print whole; long compile
    # logs get an error summary first, since the tail is usually link noise.
    if [ "$(wc -l < "$LOG")" -le 15 ]; then
      echo "    FAIL —"
      sed 's/^/      /' "$LOG"
    else
      echo "    FAIL — errors, then tail:"
      grep -E "error:|ABORT:" "$LOG" | grep -v "^oss/" | sort -u | head -10 | sed 's/^/      /'
      tail -15 "$LOG" | sed 's/^/      /'
    fi
  fi
}

skip() { RESULTS+=("SKIP  $1"); printf '::: %s\n    SKIP (%s)\n' "$1" "$2"; }

echo "############ preflight ############"
uname -sm

# Darwin needs Xcode's Metal Toolchain to build ANY addon containing GPU
# kernels. Sources with GPU code emit metallib regardless of
# --target-accelerator, so the *_ACCEL vars cannot opt out of this. Without the
# component, 9 of the 11 builds below fail identically and unhelpfully — catch
# it once, here, with the fix.
if [ "$(uname -s)" = "Darwin" ]; then
  if xcodebuild -showComponent MetalToolchain 2>/dev/null | grep -q "Status: installed"; then
    echo "Metal Toolchain: installed"
  else
    echo "FATAL: Xcode's Metal Toolchain is not installed." >&2
    echo "  Every addon with GPU kernels will fail to compile (metallib emission)." >&2
    echo "  Fix: xcodebuild -downloadComponent MetalToolchain   (~688 MB)" >&2
    exit 1
  fi
fi

# On Linux the GPU-execution suites need a real NVIDIA device. Informational
# rather than fatal: the builds and the CPU suites are still worth running on a
# GPU-less box, and the per-step status makes it obvious which ones died.
if [ "$(uname -s)" = "Linux" ]; then
  if command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
      || echo "GPU: nvidia-smi present but query failed"
  else
    echo "GPU: no nvidia-smi — the cached and retrieve suites will fail (builds still valid)"
  fi
fi

if ! command -v pixi > /dev/null 2>&1; then
  echo "FATAL: pixi not on PATH — the Mojo toolchain comes from it." >&2
  exit 1
fi
node -p "'napi-mojo ' + require('napi-mojo/package.json').version" 2>/dev/null \
  || { echo "FATAL: napi-mojo not installed — run: npm install" >&2; exit 1; }
pixi run mojo --version 2>&1 | tail -1

echo
echo "############ builds — one-shot ############"
step "build matmul"       pixi run bash examples/matmul/build.sh
step "build simd-search"  pixi run bash examples/simd-search/build.sh
step "build stats"        pixi run bash examples/stats/build.sh
step "build image"        pixi run bash examples/image/build.sh
step "build wyhash"       pixi run bash examples/wyhash/build.sh

echo
echo "############ builds — cached (persistent-buffer) ############"
step "build matmul_cached" pixi run bash examples/matmul/build_cached.sh
step "build search_cached" pixi run bash examples/simd-search/build_cached.sh
step "build stats_cached"  pixi run bash examples/stats/build_cached.sh
step "build image_cached"  pixi run bash examples/image/build_cached.sh

echo
echo "############ builds — packages ############"
# embed loads retrieve into the same process, so retrieve must build first.
step "build packages/retrieve" pixi run bash packages/retrieve/build.sh
if [ "$SKIP_EMBED" -eq 1 ]; then
  skip "build packages/embed" "--skip-embed"
else
  step "build packages/embed" pixi run bash packages/embed/build.sh
fi

echo
echo "############ correctness — one-shot (CPU paths) ############"
step "test matmul"      pixi run node examples/matmul/test.js
step "test simd-search" pixi run node examples/simd-search/test.js
step "test stats"       pixi run node examples/stats/test.js
step "test image"       pixi run node examples/image/test.js
step "test wyhash"      pixi run node examples/wyhash/test.js

echo
echo "############ correctness — cached (GPU execution) ############"
step "test matmul_cached" pixi run node examples/matmul/test_cached.js
step "test search_cached" pixi run node examples/simd-search/test_cached.js
step "test stats_cached"  pixi run node examples/stats/test_cached.js
step "test image_cached"  pixi run node examples/image/test_cached.js

echo
echo "############ correctness — packages ############"
step "jest packages/retrieve" bash -c "cd '$ROOT_DIR/packages/retrieve' && pixi run --manifest-path '$ROOT_DIR/pixi.toml' npm test"
if [ "$SKIP_EMBED" -eq 1 ]; then
  skip "test packages/embed" "--skip-embed"
elif [ "$WITH_EMBED_TEST" -eq 1 ]; then
  # test-roundtrip.js needs BOTH fixtures/ground-truth.bin and
  # fixtures/sanity-set.txt, produced together by reference.js. Neither is
  # committed (only examples/rag-demo/fixtures/*.bin is git-ignored; these are
  # simply absent), so any fresh clone — every pod, every runner — has no
  # fixtures and the test dies on "sanity-set.txt missing". That reads as an
  # embed failure and is not one; it cost a full H100 session to learn.
  #
  # Generate on demand rather than up front: reference.js pulls MiniLM through
  # @huggingface/transformers and runs 100 sentences on CPU, so it needs network
  # and a minute, and there is no reason to pay that when the fixtures exist.
  if [ -f "$ROOT_DIR/packages/embed/fixtures/ground-truth.bin" ] \
     && [ -f "$ROOT_DIR/packages/embed/fixtures/sanity-set.txt" ]; then
    skip "generate embed fixtures" "already present"
  else
    step "generate embed fixtures" pixi run node packages/embed/reference.js
  fi
  # Gate F4 correctness — embeddings match the reference within tolerance.
  step "test packages/embed" pixi run node packages/embed/test-roundtrip.js
  # The composition claim itself: embed.node (MAX Python interop) and
  # retrieve.node (Mojo N-API) loaded into ONE Node process with separate CUDA
  # contexts. test-roundtrip.js exercises embed alone, so without this the
  # kernel-factory thesis goes unverified.
  step "demo packages/embed" pixi run node packages/embed/demo.js
else
  # Downloads MiniLM weights and needs a working Accelerator(); on Apple silicon
  # MAX reports "Not implemented for device: Apple M4", so this is opt-in.
  skip "test packages/embed" "opt in with --with-embed-test"
  skip "demo packages/embed" "opt in with --with-embed-test"
fi

echo
echo "############ summary ############"
printf '%s\n' "${RESULTS[@]}"
echo
if [ "$FAILED" -eq 0 ]; then echo "ALL GREEN"; else echo "SOME FAILURES — see above"; fi
exit "$FAILED"
