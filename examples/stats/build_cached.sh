#!/usr/bin/env bash
# Build Phase 3b.2 cached stats addon: persistent device buffers + Welford-ready kernels
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
NAPI_SRC="$ROOT_DIR/node_modules/napi-mojo/src"

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

# GPU target: Darwin arm64 → metal:4, Linux x86_64 → sm_90 (H100/H200).
# Override with STATS_ACCEL="" or STATS_ACCEL="--target-accelerator sm_80" etc.
ACCEL_FLAG="${STATS_ACCEL-}"
if [ -z "${STATS_ACCEL+x}" ]; then
    if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        ACCEL_FLAG="--target-accelerator metal:4"
    elif [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
        ACCEL_FLAG="--target-accelerator sm_90"
    fi
fi

mojo build --emit shared-lib ${MCPU_FLAG} ${ACCEL_FLAG} -I "$NAPI_SRC" \
    "$SCRIPT_DIR/addon_cached.mojo" -o "$SCRIPT_DIR/build/stats_cached.${LIB_EXT}"

mv "$SCRIPT_DIR/build/stats_cached.${LIB_EXT}" "$SCRIPT_DIR/build/stats_cached.node"

echo "Build complete: stats/build/stats_cached.node"
