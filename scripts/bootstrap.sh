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
# Discard any local modifications first — pod disk is ephemeral, and
# `pixi install` / build steps routinely touch lockfiles that would
# otherwise block the checkout.
git reset --hard HEAD 2>/dev/null || true

# Which ref to run. Defaults to main; override with REPO_REF=some/branch for a
# session that needs unmerged work.
#
# This used to hardcode `spike/embedding-kernel`, from the 2026-04 embedding
# spike. That branch was deleted from origin after the spike was productized
# into packages/embed, so the fallback silently took over — leaving pods
# running main's code while *reporting* "on spike/embedding-kernel", a string
# that then lands in committed capture files. The follow-up `git reset --hard
# origin/$(current branch)` also resolved to the nonexistent
# origin/spike/embedding-kernel, failed, and was swallowed by `|| true`, so the
# belt-and-suspenders check had quietly stopped checking anything.
REPO_REF="${REPO_REF:-main}"
if ! git ls-remote --exit-code origin "$REPO_REF" >/dev/null 2>&1; then
  echo "FATAL: origin/$REPO_REF does not exist — refusing to run against an unknown ref." >&2
  exit 1
fi
# -B (not -b) handles a prior session having left a stale local branch of the
# same name; plain `git checkout` would succeed on it and never see origin.
git checkout -B "$REPO_REF" "origin/$REPO_REF"
# Belt-and-suspenders: ensure the tree matches origin exactly. Unlike before,
# this ref is known to exist, so a failure here is a real failure.
git reset --hard "origin/$REPO_REF"

echo "Ready. cd $REPO (on $(git rev-parse --abbrev-ref HEAD) at $(git rev-parse --short HEAD))"
