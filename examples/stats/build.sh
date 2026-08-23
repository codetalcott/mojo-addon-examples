#!/usr/bin/env bash
# Build stats addon: compile Mojo -> .node shared library
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$ROOT_DIR/scripts/napi-include.sh"

mkdir -p "$SCRIPT_DIR/build"

case "$(uname -s)" in
    Darwin) LIB_EXT="dylib" ;;
    Linux)  LIB_EXT="so" ;;
    *)      echo "Unsupported platform: $(uname -s)" >&2; exit 1 ;;
esac

MCPU_FLAG=""
if [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
    MCPU_FLAG="--mcpu haswell"
fi

# GPU target: Mojo needs --target-accelerator for heterogeneous compilation.
# Defaults:
#   Darwin arm64  → metal:4  (covers M1-M4)
#   Linux         → sm_90    (NVIDIA Hopper — H100/H200, and GH200, which is
#                             aarch64 + Hopper). Not gated on x86_64: the flag names
#                             the NVIDIA target, not the host CPU. Without it Mojo
#                             falls back to host detection and dies with "Unknown GPU
#                             architecture detected" wherever no GPU is present.
# Override with a specific flag like STATS_ACCEL="--target-accelerator sm_80" for
# A100, "sm_100a" for B100/B200, etc. Note STATS_ACCEL="" does NOT give a CPU-only
# build — it only drops the flag, and Mojo then falls back to the *host* accelerator.
# There is no "no accelerator" flag; see CLAUDE.md "GPU target flags".
ACCEL_FLAG="${STATS_ACCEL-}"
if [ -z "${STATS_ACCEL+x}" ]; then
    if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        ACCEL_FLAG="--target-accelerator metal:4"
    elif [ "$(uname -s)" = "Linux" ]; then
        ACCEL_FLAG="--target-accelerator sm_90"
    fi
fi

mojo build --emit shared-lib ${MCPU_FLAG} ${ACCEL_FLAG} -I "$NAPI_SRC" \
    "$SCRIPT_DIR/addon.mojo" -o "$SCRIPT_DIR/build/stats.${LIB_EXT}"

mv "$SCRIPT_DIR/build/stats.${LIB_EXT}" "$SCRIPT_DIR/build/stats.node"

echo "Build complete: stats/build/stats.node"
