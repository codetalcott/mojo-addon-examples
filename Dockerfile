# Dockerfile — Multi-stage build for mojo-addon-examples demo
#
# Stage 1: Build Mojo addons (needs pixi + Mojo nightly)
# Stage 2: Slim Node.js runtime with pre-built .node files

# --- Stage 1: Builder -------------------------------------------------------

FROM ubuntu:22.04 AS builder

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js 20
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

# Copy project files
WORKDIR /app
COPY pixi.toml pixi.lock ./
COPY package.json package-lock.json ./

# Install dependencies
RUN pixi install
RUN npm ci

# Copy source files
COPY matmul/ matmul/
COPY simd-search/ simd-search/
COPY stats/ stats/
COPY image/ image/
COPY wyhash/ wyhash/

# Build all addons
RUN pixi run bash -c "npm run build:all"

# --- Stage 2: Runtime --------------------------------------------------------

FROM node:20-slim

WORKDIR /app

# Copy demo package and install
COPY demo/package.json demo/package-lock.json* demo/
RUN cd demo && npm ci --omit=dev 2>/dev/null || cd demo && npm install --omit=dev

# Copy pre-built .node addon files from builder
COPY --from=builder /app/matmul/build/matmul.node matmul/build/
COPY --from=builder /app/simd-search/build/search.node simd-search/build/
COPY --from=builder /app/stats/build/stats.node stats/build/
COPY --from=builder /app/image/build/image.node image/build/
COPY --from=builder /app/wyhash/build/wyhash.node wyhash/build/

# Copy JS files that addons are loaded through (require paths)
COPY matmul/*.js matmul/
COPY simd-search/*.js simd-search/
COPY stats/*.js stats/
COPY image/*.js image/
COPY wyhash/*.js wyhash/

# Copy demo source
COPY demo/ demo/

EXPOSE 8080
CMD ["node", "demo/server.js"]
