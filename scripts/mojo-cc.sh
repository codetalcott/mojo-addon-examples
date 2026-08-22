#!/usr/bin/env bash
# Compiler shim for `napi-mojo build`.
#
# The CLI builds every addon with a fixed command line and offers no way to
# append flags to `mojo build`. These addons need one: on x86-64 Linux,
# --mcpu haswell enables AVX2. Without it the backend targets baseline x86-64
# (SSE2) and every Float64 SIMD loop runs 2 lanes wide instead of 4 -- confirmed
# with `mojo build --emit asm --target-triple x86_64-unknown-linux-gnu`, which
# emits xmm and no vaddpd without the flag, ymm and vaddpd with it.
#
# napi-mojo invokes this via `--mojo scripts/mojo-cc.sh`, so it receives the
# whole `build --emit shared-lib -I ... <entry> -o <out>` argument list and just
# forwards it with the extra flags appended.
set -eo pipefail

flags=()
if [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
    flags+=(--mcpu haswell)
fi

# Inside `pixi run` -- how CI and the Dockerfile build -- mojo is already on
# PATH and is the version pinned in pixi.toml. Fall back to pixi otherwise so a
# bare `npm run build:all` still uses the pinned toolchain rather than whatever
# mojo happens to be installed.
if command -v mojo >/dev/null 2>&1; then
    exec mojo "$@" "${flags[@]}"
fi
exec pixi run mojo "$@" "${flags[@]}"
