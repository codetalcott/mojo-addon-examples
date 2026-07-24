#!/usr/bin/env bash
# Regenerate NOTICE files in each platform sub-package from NOTICE.template,
# using the MAX/Mojo versions actually installed in the pixi env. Sourcing
# from .pixi/envs/default/conda-meta/ guarantees the notice names the
# exact versions that will ship in build/gpu-libs/ — it cannot drift from
# the pin in pixi.toml.
#
# Run before `npm publish`:
#   pixi run bash packages/retrieve/scripts/generate-notice.sh
# or (from packages/retrieve/):
#   npm run notice
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$PKG_DIR/../.." && pwd)"
TEMPLATE="$PKG_DIR/NOTICE.template"
CONDA_META="$ROOT_DIR/.pixi/envs/default/conda-meta"

[ -f "$TEMPLATE" ] || { echo "missing $TEMPLATE"; exit 1; }
[ -d "$CONDA_META" ] || { echo "pixi env not installed at $CONDA_META — run: (cd $ROOT_DIR && pixi install)"; exit 1; }

# conda-meta filenames: <name>-<version>-<build>.json
# "max-[0-9]*" matches the 'max' package only (not 'max-core').
shopt -s nullglob
max_candidates=("$CONDA_META"/max-[0-9]*.json)
mojo_candidates=("$CONDA_META"/mojo-compiler-[0-9]*.json)
shopt -u nullglob

[ "${#max_candidates[@]}" -gt 0 ] || { echo "MAX not installed — cannot determine version"; exit 1; }
[ "${#mojo_candidates[@]}" -gt 0 ] || { echo "mojo-compiler not installed — cannot determine version"; exit 1; }

max_base=$(basename "${max_candidates[0]}" .json)
mojo_base=$(basename "${mojo_candidates[0]}" .json)

MAX_VERSION="${max_base#max-}";           MAX_VERSION="${MAX_VERSION%-*}"
MOJO_VERSION="${mojo_base#mojo-compiler-}"; MOJO_VERSION="${MOJO_VERSION%-*}"

for platform in darwin-arm64 linux-x64; do
  out="$PKG_DIR/npm/$platform/NOTICE"
  sed -e "s|{PLATFORM}|$platform|g" \
      -e "s|{MAX_VERSION}|$MAX_VERSION|g" \
      -e "s|{MOJO_VERSION}|$MOJO_VERSION|g" \
      "$TEMPLATE" > "$out"
  echo "wrote $out  (MAX=$MAX_VERSION  Mojo=$MOJO_VERSION)"
done
