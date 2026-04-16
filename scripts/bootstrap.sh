#!/usr/bin/env bash
# scripts/bootstrap.sh — canonical source for the pod-side session bootstrap.
#
# This file is the source of truth. The Network Volume holds a copy at
# /workspace/persist/bootstrap.sh which pods source at session start. To
# update an existing volume, run:
#
#   ./scripts/runpod-launch.sh -- \
#     "cd /workspace/mojo-addon-examples && git checkout -B spike/embedding-kernel origin/spike/embedding-kernel && cp scripts/bootstrap.sh /workspace/persist/bootstrap.sh && chmod +x /workspace/persist/bootstrap.sh"
#
# Fresh seeds should cp this file onto the volume during step 4 of the
# Day 0 setup (see ideas/embedding-kernel-spike-plan.md).
#
# What it does:
#   - Sources secrets (GH_TOKEN)
#   - Adds volume-local pixi/gh/npm bins to PATH
#   - Points pixi + HF caches at the Network Volume
#   - Clones the repo (first run) or pulls latest
#   - Force-resets the spike branch to origin (handles the case where a
#     prior session left a stale local branch tracking main)

set -euo pipefail

source /workspace/persist/secrets/github.env
export PATH="/workspace/persist/bin:/workspace/persist/pixi-home/bin:$PATH"
export PIXI_CACHE_DIR=/workspace/persist/pixi-cache
export HF_HOME=/workspace/persist/model-cache

gh auth setup-git

REPO=/workspace/mojo-addon-examples
if [ ! -d "$REPO" ]; then
  gh repo clone codetalcott/mojo-addon-examples "$REPO"
fi
cd "$REPO"

git fetch origin
# Always force-reset to origin's spike branch. Using -B (not -b) handles the
# case where a prior session's `git checkout -b` created a local branch
# tracking main — plain `git checkout spike/...` would silently succeed on
# the stale local branch and never pull in origin's latest commits.
if git ls-remote --exit-code origin spike/embedding-kernel >/dev/null 2>&1; then
  git checkout -B spike/embedding-kernel origin/spike/embedding-kernel
else
  git checkout -B spike/embedding-kernel origin/main
fi

echo "Ready. cd $REPO (on $(git rev-parse --abbrev-ref HEAD) at $(git rev-parse --short HEAD))"
