#!/usr/bin/env bash
# Build script for the embedding-kernel spike.
# Compiles spike/src/lib.mojo into spike/build/embed.node.
# Mirrors packages/rag/build.sh — same pixi env, same napi-mojo framework.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NAPI_SRC="$ROOT_DIR/node_modules/napi-mojo/src"

mkdir -p "$SCRIPT_DIR/build"

if [ ! -d "$NAPI_SRC" ]; then
    echo "napi-mojo not installed in repo root — run: (cd $ROOT_DIR && npm install)" >&2
    exit 1
fi

case "$(uname -s)" in
    Darwin) LIB_EXT="dylib" ;;
    Linux)  LIB_EXT="so" ;;
    *)      echo "Unsupported platform: $(uname -s)" >&2; exit 1 ;;
esac

MCPU_FLAG=""
if [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
    MCPU_FLAG="--mcpu haswell"
fi

ACCEL_FLAG="${SPIKE_EMBED_ACCEL-}"
if [ -z "${SPIKE_EMBED_ACCEL+x}" ]; then
    if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        ACCEL_FLAG="--target-accelerator metal:4"
    elif [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
        ACCEL_FLAG="--target-accelerator sm_80"
    fi
fi

mojo build --emit shared-lib ${MCPU_FLAG} ${ACCEL_FLAG} \
    -I "$SCRIPT_DIR/src" \
    -I "$NAPI_SRC" \
    "$SCRIPT_DIR/src/lib.mojo" \
    -o "$SCRIPT_DIR/build/libembed.${LIB_EXT}"

mv "$SCRIPT_DIR/build/libembed.${LIB_EXT}" "$SCRIPT_DIR/build/embed.node"

echo "Build complete: spike/build/embed.node"
