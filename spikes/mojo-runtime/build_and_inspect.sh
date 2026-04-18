#!/usr/bin/env bash
# Build each tier{0..4} as a Mojo shared library, then capture both direct
# (otool -L) and transitive (otool -L recursive walk) dynamic-library
# dependencies into spikes/mojo-runtime/build/<tier>.deps.txt.
#
# Run from repo root: pixi run bash spikes/mojo-runtime/build_and_inspect.sh
#
# On Linux, swap otool for ldd in the inspect_* functions; the build half is
# identical except --target-accelerator changes (sm_80 vs metal:4) and the
# library extension (so vs dylib).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"

case "$(uname -s)" in
    Darwin) LIB_EXT="dylib" ;;
    Linux)  LIB_EXT="so" ;;
    *)      echo "Unsupported platform: $(uname -s)" >&2; exit 1 ;;
esac

MCPU_FLAG=""
if [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
    MCPU_FLAG="--mcpu haswell"
fi

# GPU target — only applied to tier2+ (CPU-only tiers compile without it).
GPU_ACCEL=""
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    GPU_ACCEL="--target-accelerator metal:4"
elif [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
    GPU_ACCEL="--target-accelerator sm_80"
fi

list_deps() {
    # Print direct dynamic deps of a single binary, one per line, no header.
    # Darwin returns @rpath/libFoo.dylib or absolute paths; Linux returns
    # absolute resolved paths (via the ldd "name => path" split) or the name
    # if no resolution (vdso etc).
    local bin="$1"
    if [ "$(uname -s)" = "Darwin" ]; then
        otool -L "$bin" | tail -n +2 | awk '{print $1}'
    else
        ldd "$bin" 2>/dev/null | awk '
          $2 == "=>" { print $3; next }
          NF >= 1    { print $1 }
        '
    fi
}

walk_deps() {
    # Transitive walk: BFS over dependencies, skipping libSystem/system paths.
    local bin="$1"
    local out="$2"
    local seen=""
    local queue="$bin"
    : > "$out"
    while [ -n "$queue" ]; do
        local next=""
        for b in $queue; do
            for dep in $(list_deps "$b"); do
                # Skip self-references and stuff we already have
                case " $seen " in
                    *" $dep "*) continue ;;
                esac
                seen="$seen $dep"
                echo "$dep" >> "$out"
                # Only recurse into deps inside our pixi env (where MAX libs live)
                # and our build dir; skip /usr/lib system libs.
                case "$dep" in
                    "$ROOT_DIR/.pixi/"*) next="$next $dep" ;;
                    "$BUILD_DIR/"*) next="$next $dep" ;;
                    "@rpath/"*)
                        # Resolve @rpath against pixi lib dir (the rpath we'll set)
                        local resolved="$ROOT_DIR/.pixi/envs/default/lib/${dep#@rpath/}"
                        if [ -f "$resolved" ]; then next="$next $resolved"; fi
                        ;;
                esac
            done
        done
        queue="$next"
    done
}

build_tier() {
    local tier="$1"; local src="$2"; local accel="$3"
    local out="$BUILD_DIR/${tier}.${LIB_EXT}"
    echo "==> building $tier ($src)"
    pixi run mojo build --emit shared-lib $MCPU_FLAG $accel "$src" -o "$out"
    # Set rpath so transitive walks resolve @rpath/libFoo against pixi lib
    if [ "$(uname -s)" = "Darwin" ]; then
        install_name_tool -add_rpath "$ROOT_DIR/.pixi/envs/default/lib" "$out" 2>/dev/null || true
    fi
    echo "    direct deps:"
    list_deps "$out" | sed 's/^/      /'
    walk_deps "$out" "$BUILD_DIR/${tier}.deps.txt"
    echo "    transitive deps -> $BUILD_DIR/${tier}.deps.txt ($(wc -l < "$BUILD_DIR/${tier}.deps.txt") entries)"
    echo
}

build_tier tier0 "$SRC_DIR/tier0_cpu_only.mojo"   ""
build_tier tier1 "$SRC_DIR/tier1_cpu_simd.mojo"   ""
build_tier tier2 "$SRC_DIR/tier2_gpu_raw.mojo"    "$GPU_ACCEL"
build_tier tier3 "$SRC_DIR/tier3_gpu_layout.mojo" "$GPU_ACCEL"
build_tier tier4 "$SRC_DIR/tier4_gpu_linalg.mojo" "$GPU_ACCEL"

echo "==> all builds done"
