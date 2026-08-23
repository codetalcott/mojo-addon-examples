# Dockerfile — Multi-stage build for the mojo-addon-examples demo
#
# Stage 1: build the Mojo addons (needs pixi + the Mojo toolchain)
# Stage 2: slim Node runtime carrying only the built addons
#
# Two things about this build are easy to get wrong and expensive to discover:
#
#  1. A napi-mojo .node is NOT self-contained. It carries hard @rpath/RUNPATH
#     dependencies on the Mojo runtime (libKGENCompilerRTShared,
#     libAsyncRTMojoBindings, ...) that only resolve inside the pixi env. Copy
#     the bare .node into a slim image and it dies at require() with
#     "libKGENCompilerRTShared.so: cannot open shared object file"
#     (ERR_DLOPEN_FAILED). scripts/bundle-runtime.sh from napi-mojo walks the
#     real dependency closure and rewrites RUNPATH to $ORIGIN; the runtime stage
#     then copies the WHOLE build directory, because copying just the .node
#     orphans the sibling libraries and reproduces the failure exactly.
#     The closure is platform-dependent, so it is computed, never hardcoded.
#
#  2. Local `npm test` cannot catch (1) — it runs with the pixi env still on the
#     library search path. Only running the container proves it.

# --- Stage 1: Builder -------------------------------------------------------

FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# patchelf is what bundle-runtime.sh uses to read and rewrite RUNPATH on Linux;
# it is not in the base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git build-essential patchelf && \
    rm -rf /var/lib/apt/lists/*

# Node 22: napi-mojo 0.13 resolves N-API symbols at load time that Node 20 does
# not export, so require() throws there. Matches engines.node in package.json.
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

WORKDIR /app

# pixi.lock is git-ignored in this repo, so it may or may not be present. The
# bracket glob copies it when it exists and is a no-op when it does not, rather
# than failing the build on a fresh clone.
COPY pixi.toml pixi.loc[k] ./
RUN pixi install

# The root package.json declares file: dependencies on both workspace packages,
# so their manifests have to exist before npm ci will resolve.
COPY package.json package-lock.json ./
COPY packages/retrieve/package.json packages/retrieve/
COPY packages/embed/package.json packages/embed/
RUN npm ci

COPY scripts/ scripts/
COPY examples/ examples/

# Build the five one-shot addons, then make each self-contained. build:all does
# not bundle — bundling is a packaging step, and doing it here keeps local
# builds fast while still giving the runtime stage something that loads.
RUN pixi run bash -c "npm run build:all"
RUN for n in examples/matmul/build/matmul.node \
             examples/simd-search/build/search.node \
             examples/stats/build/stats.node \
             examples/image/build/image.node \
             examples/wyhash/build/wyhash.node; do \
        echo "== bundling $n" && \
        pixi run bash node_modules/napi-mojo/scripts/bundle-runtime.sh "$n"; \
    done

# Fail the build here rather than at container start if anything still points
# outside the image. ldd exits non-zero on "not found" entries.
RUN for n in examples/*/build/*.node; do \
        echo "== checking $n" && \
        ! ldd "$n" 2>/dev/null | grep -q "not found" || \
            { echo "UNRESOLVED DEPENDENCY in $n:"; ldd "$n" | grep "not found"; exit 1; }; \
    done

# --- Stage 2: Runtime --------------------------------------------------------

FROM node:22-slim

WORKDIR /app

COPY demo/package.json demo/package-lock.json demo/
RUN cd demo && (npm ci --omit=dev || npm install --omit=dev)

# Whole build directories, not just the .node files: the bundled Mojo runtime
# libraries sit beside each addon and are found through RUNPATH=$ORIGIN.
COPY --from=builder /app/examples/matmul/build/ examples/matmul/build/
COPY --from=builder /app/examples/simd-search/build/ examples/simd-search/build/
COPY --from=builder /app/examples/stats/build/ examples/stats/build/
COPY --from=builder /app/examples/image/build/ examples/image/build/
COPY --from=builder /app/examples/wyhash/build/ examples/wyhash/build/

# JS the demo loads the addons through
COPY examples/matmul/*.js examples/matmul/
COPY examples/simd-search/*.js examples/simd-search/
COPY examples/stats/*.js examples/stats/
COPY examples/image/*.js examples/image/
COPY examples/wyhash/*.js examples/wyhash/

COPY demo/server.js demo/
COPY demo/public/ demo/public/
COPY demo/assets/ demo/assets/

# Prove the addons load before the image is considered good. Without this a
# broken bundle only shows up as a crash on first request in production.
RUN node -e "for (const a of ['matmul/build/matmul','simd-search/build/search','stats/build/stats','image/build/image','wyhash/build/wyhash']) { require('/app/examples/' + a + '.node'); } console.log('all addons load')"

EXPOSE 8080
CMD ["node", "demo/server.js"]
