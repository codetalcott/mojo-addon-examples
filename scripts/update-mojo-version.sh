#!/usr/bin/env bash
# Update the pinned MAX/Mojo nightly in pixi.toml
# Usage: ./scripts/update-mojo-version.sh 26.5.0.dev2026072306
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <max-version>"
  echo "Example: $0 26.5.0.dev2026072306"
  exit 1
fi

NEW_VERSION="$1"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PIXI_TOML="$ROOT_DIR/pixi.toml"

# The pin lives on the `max` dependency, not `mojo` — the Mojo toolchain ships
# inside the MAX conda package. An earlier version of this script targeted
# `mojo = "=="`, which stopped matching when the pin moved and silently no-oped.
# Fail loudly instead of pretending to have updated anything.
if ! grep -qE '^max = "' "$PIXI_TOML"; then
  echo "error: no 'max = \"...\"' pin found in $PIXI_TOML" >&2
  exit 1
fi

sed -i.bak -E "s|^max = \".*\"|max = \"==$NEW_VERSION\"|" "$PIXI_TOML"
rm -f "$PIXI_TOML.bak"

if ! grep -qF "max = \"==$NEW_VERSION\"" "$PIXI_TOML"; then
  echo "error: pin rewrite failed" >&2
  exit 1
fi

echo "Updated pixi.toml to max == $NEW_VERSION"
pixi install
