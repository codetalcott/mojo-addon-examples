#!/usr/bin/env bash
# Build script for @qkstat/retrieve
# Compiles src/lib.mojo into build/retrieve.node, pulling napi-mojo's N-API framework
# from the monorepo's installed napi-mojo (see scripts/napi-include.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$ROOT_DIR/scripts/napi-include.sh"

mkdir -p "$SCRIPT_DIR/build"

# Detect platform-specific shared library extension
case "$(uname -s)" in
    Darwin) LIB_EXT="dylib" ;;
    Linux)  LIB_EXT="so" ;;
    *)      echo "Unsupported platform: $(uname -s)" >&2; exit 1 ;;
esac

# On Linux x86_64, target Haswell (2013) to avoid AVX-512 instructions
# that aren't available on GitHub Actions runners.
MCPU_FLAG=""
if [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
    MCPU_FLAG="--mcpu haswell"
fi

# GPU target: Darwin arm64 → metal:4, Linux x86_64 → sm_80 (NVIDIA baseline;
# PTX forward-compat covers sm_80/86/89/90/100+ via driver JIT — one binary
# ships to all NVIDIA users). Override with QKSTAT_RETRIEVE_ACCEL="--target-accelerator sm_90"
# for a native sm_90 build (Hopper-specific wgmma/TMA — future optimization).
ACCEL_FLAG="${QKSTAT_RETRIEVE_ACCEL-}"
if [ -z "${QKSTAT_RETRIEVE_ACCEL+x}" ]; then
    if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        ACCEL_FLAG="--target-accelerator metal:4"
    elif [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
        ACCEL_FLAG="--target-accelerator sm_80"
    fi
fi

# `-I src` puts packages/retrieve/src on the include path so lib.mojo can
# `from linalg import ...`.  `-I $NAPI_SRC` resolves `napi.*` packages.
mojo build --emit shared-lib ${MCPU_FLAG} ${ACCEL_FLAG} \
    -I "$SCRIPT_DIR/src" \
    -I "$NAPI_SRC" \
    "$SCRIPT_DIR/src/lib.mojo" \
    -o "$SCRIPT_DIR/build/libretrieve.${LIB_EXT}"

mv "$SCRIPT_DIR/build/libretrieve.${LIB_EXT}" "$SCRIPT_DIR/build/retrieve.node"

echo "Build complete: packages/retrieve/build/retrieve.node"
