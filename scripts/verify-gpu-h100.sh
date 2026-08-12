#!/usr/bin/env bash
# scripts/verify-gpu-h100.sh — verify every build target on a real NVIDIA GPU.
#
# Written for the dev2026072306 migration, whose five GPU-bearing targets could
# not be verified on an M4 laptop: files containing GPU kernels emit metallib on
# Darwin regardless of --target-accelerator, and that needs Xcode's Metal
# Toolchain. On Linux x86_64 the build scripts default to sm_90, so this is the
# first place the migration's GPU paths actually compile.
#
# Deliberately does NOT use `set -e`. This is a survey: one failing target must
# not hide the other seven. Each step reports its own status and the script
# exits non-zero only if something actually failed.
#
# Usage (from repo root on the pod):
#   bash scripts/verify-gpu-h100.sh [git-ref]

set -uo pipefail

REF="${1:-fix/mojo-nightly-2026072306}"
FAILED=0
declare -a RESULTS=()

step() {  # step <label> <cmd...>
  local label="$1"; shift
  echo "::: $label"
  if "$@" > /tmp/step.log 2>&1; then
    RESULTS+=("PASS  $label")
    echo "    PASS"
  else
    RESULTS+=("FAIL  $label")
    FAILED=1
    echo "    FAIL — last 25 lines:"
    grep -E "error:" /tmp/step.log | grep -v "^oss/" | sort -u | head -15 | sed 's/^/      /'
    tail -25 /tmp/step.log | sed 's/^/      /'
  fi
}

echo "############ environment ############"
uname -a
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

echo
echo "############ sync repo to $REF ############"
git fetch origin --prune
git checkout -B "$REF" "origin/$REF" || { echo "FATAL: cannot check out origin/$REF"; exit 1; }
git log --oneline -1

echo
echo "############ toolchain ############"
step "npm install"  npm install --no-audit --no-fund
step "pixi install" pixi install
echo "napi-mojo: $(node -e 'console.log(require("./node_modules/napi-mojo/package.json").version)' 2>/dev/null || echo unknown)"
pixi run mojo --version 2>&1 | tail -1

echo
echo "############ builds — GPU-bearing (sm_90) ############"
step "build matmul_cached" pixi run bash examples/matmul/build_cached.sh
step "build stats"         pixi run bash examples/stats/build.sh
step "build stats_cached"  pixi run bash examples/stats/build_cached.sh
step "build image"         pixi run bash examples/image/build.sh
step "build image_cached"  pixi run bash examples/image/build_cached.sh
step "build simd-search"   pixi run bash examples/simd-search/build.sh
step "build search_cached" pixi run bash examples/simd-search/build_cached.sh
step "build packages/retrieve"  pixi run bash packages/retrieve/build.sh

echo
echo "############ builds — CPU-only ############"
step "build matmul" pixi run bash examples/matmul/build.sh
step "build wyhash" pixi run bash examples/wyhash/build.sh

echo
echo "############ correctness ############"
step "npm test"              pixi run npm test
step "test search_cached"    pixi run node examples/simd-search/test_cached.js
step "jest packages/retrieve"     bash -c "cd packages/retrieve && pixi run --manifest-path ../../pixi.toml npm test"

echo
echo "############ parallelize() thread dispatch ############"
# napi-mojo 0.6.0 restored the AsyncRT entry point. Under the old sequential
# fallback this ratio sat at ~1.0x; on M4 after the fix it measured 2.62x.
pixi run node -e '
const m = require("./examples/matmul/build/matmul.node");
const N = 512;
const A = new Float64Array(N*N).fill(1.5), B = new Float64Array(N*N).fill(2.5), C = new Float64Array(N*N);
const bench = (fn, reps=5) => { let best=Infinity; for (let i=0;i<reps;i++){ const t=process.hrtime.bigint(); m[fn](A,B,C,N,N,N); const d=Number(process.hrtime.bigint()-t)/1e6; if(d<best)best=d; } return best; };
const tiled = bench("matmulTiled"), par = bench("matmulParallel");
console.log("  matmulTiled   :", tiled.toFixed(1), "ms");
console.log("  matmulParallel:", par.toFixed(1), "ms");
console.log("  speedup       :", (tiled/par).toFixed(2)+"x", tiled/par > 1.5 ? "-> threads dispatching" : "-> SEQUENTIAL");
' 2>&1 || echo "  parallel benchmark failed"

echo
echo "############ summary ############"
printf '%s\n' "${RESULTS[@]}"
echo
if [ "$FAILED" -eq 0 ]; then echo "ALL GREEN"; else echo "SOME FAILURES — see above"; fi
exit "$FAILED"
