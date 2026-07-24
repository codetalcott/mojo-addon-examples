#!/usr/bin/env bash
# Post-build rpath bundling — Linux only. Runs on CI after build.sh to
# produce a self-contained Linux prebuilt: retrieve.node + gpu-libs/*.so with
# rpath set to $ORIGIN/gpu-libs so end users don't need pixi / MAX / CUDA
# toolkit on their machine (just the NVIDIA driver, which every GPU host
# has).
#
# Gated on Modular Community License sign-off — see the root CLAUDE.md
# or project memory for status. Source build (via build.sh) does not
# require this step.
set -euo pipefail

[ "$(uname -s)" = "Linux" ] || { echo "bundle-libs.sh: skipping — not Linux"; exit 0; }
which patchelf >/dev/null || { echo "install patchelf: apt-get install -y patchelf"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$PKG_DIR/../.." && pwd)"
PIXI_LIB="$ROOT_DIR/.pixi/envs/default/lib"

[ -d "$PIXI_LIB" ] || { echo "pixi env not installed at $PIXI_LIB — run: (cd $ROOT_DIR && pixi install)"; exit 1; }
[ -f "$PKG_DIR/build/retrieve.node" ] || { echo "build/retrieve.node missing — run: bash build.sh first"; exit 1; }

LIBS_DIR="$PKG_DIR/build/gpu-libs"
mkdir -p "$LIBS_DIR"

# Seven libs identified by the spike on H100 RunPod (see memory project
# notes / napi-mojo spike/gpu-fatbin/FINDINGS.md). Zero direct CUDA runtime
# deps — libNVPTX.so is Mojo's own driver wrapper that dlopens libcuda.so.1
# at runtime.
LIBS=(
  libKGENCompilerRTShared.so
  libAsyncRTMojoBindings.so
  libAsyncRTRuntimeGlobals.so
  libMSupportGlobals.so
  libNVPTX.so
  libstdc++.so.6
  libgcc_s.so.1
)

for lib in "${LIBS[@]}"; do
  src="$PIXI_LIB/$lib"
  [ -f "$src" ] || { echo "missing expected lib: $src"; exit 1; }
  cp -L "$src" "$LIBS_DIR/"
done

patchelf --set-rpath '$ORIGIN/gpu-libs' "$PKG_DIR/build/retrieve.node"

echo "Bundled $(ls "$LIBS_DIR" | wc -l | tr -d ' ') libs into $LIBS_DIR/"
echo "rpath on retrieve.node: $(patchelf --print-rpath "$PKG_DIR/build/retrieve.node")"
