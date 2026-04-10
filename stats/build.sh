#!/usr/bin/env bash
# Build stats addon: compile Mojo -> .node shared library
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
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

# GPU target: Mojo needs --target-accelerator for heterogeneous compilation.
# Darwin arm64 → metal:4 (covers M1-M4). Override with STATS_ACCEL="" to
# build CPU-only (e.g. on a box without a GPU toolchain).
ACCEL_FLAG="${STATS_ACCEL-}"
if [ -z "${STATS_ACCEL+x}" ]; then
    if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        ACCEL_FLAG="--target-accelerator metal:4"
    fi
fi

mojo build --emit shared-lib ${MCPU_FLAG} ${ACCEL_FLAG} -I "$NAPI_SRC" \
    "$SCRIPT_DIR/addon.mojo" -o "$SCRIPT_DIR/build/stats.${LIB_EXT}"

mv "$SCRIPT_DIR/build/stats.${LIB_EXT}" "$SCRIPT_DIR/build/stats.node"

echo "Build complete: stats/build/stats.node"
