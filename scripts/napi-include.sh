#!/usr/bin/env bash
# Resolve napi-mojo's Mojo include directory — the path every addon build
# passes to `mojo build -I` so that `from napi.framework...` resolves.
#
# napi-mojo 0.7+ is a source framework whose npm entry exports paths rather
# than a compiled addon (the node-addon-api model), so `.include` is the
# supported way to locate the framework sources. Resolving through node also
# survives npm workspace hoisting and `npm link`, which a hardcoded
# node_modules/napi-mojo/src does not.
#
# Usage (from a build.sh, with ROOT_DIR already set to the repo root):
#     source "$ROOT_DIR/scripts/napi-include.sh"
#     # $NAPI_SRC is now set, or the script has exited with a message.

NAPI_SRC="$(cd "$ROOT_DIR" && node -p "require('napi-mojo').include" 2>/dev/null || true)"

if [ -z "$NAPI_SRC" ] || [ ! -d "$NAPI_SRC" ]; then
    echo "napi-mojo not installed in repo root — run: (cd $ROOT_DIR && npm install)" >&2
    exit 1
fi
