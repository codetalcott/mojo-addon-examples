#!/usr/bin/env bash
# scripts/runpod-launch.sh — one-shot RunPod bench runner.
#
# Sibling of lambda-bench.sh with the same contract: launch a GPU pod with a
# persistent Network Volume attached, source the repo's bootstrap script
# (secrets + pixi + gh + git clone), run a command inside mojo-addon-examples/,
# capture output, and terminate on exit regardless of how the script exits.
#
# Uses RunPod's GraphQL API (https://api.runpod.io/graphql). Verify the
# mutation shapes against your current API docs — RunPod has been iterating
# on the schema. The --dry-run flag prints payloads without calling the API.
#
# Prerequisites (one-time RunPod setup):
#   - Create a Network Volume (dashboard → Storage → Network Volumes).
#     Pass its ID via --volume-id or RUNPOD_VOLUME_ID.
#   - Seed the volume with /workspace/persist/{secrets/github.env, bootstrap.sh,
#     pixi-home, bin/gh}. Mount path inside the pod is /workspace.
#   - Have an SSH public key on your laptop (~/.ssh/id_ed25519.pub or similar).
#     Its contents are injected as PUBLIC_KEY env var; the standard RunPod
#     images read this and write to /root/.ssh/authorized_keys at boot.
#
# Auth: RUNPOD_API_KEY env var, or ~/.config/runpod/env sourced on start.
#
# Usage:
#   export RUNPOD_API_KEY=...
#   export RUNPOD_VOLUME_ID=vol_xxxx
#   scripts/runpod-launch.sh [options] -- "command to run on pod"
#
# Options:
#   --gpu-type NAME           Default: "NVIDIA H100 80GB HBM3"
#                             Check current catalog via RunPod dashboard.
#   --cloud-type TYPE         SECURE | COMMUNITY   Default: SECURE
#                             Secure gets direct public IP; Community uses proxy.
#   --volume-id ID            Network volume to attach (or RUNPOD_VOLUME_ID env)
#   --no-volume               Run with no network volume: bigger container disk and a
#                             cold bootstrap (pixi + node + clone) instead of the seeded
#                             /workspace/persist. Slower to start and nothing is cached
#                             between runs, but needs no volume to exist. Only works for
#                             a public repo and workloads that need no cached weights.
#   --image NAME              Container image. Default: runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204
#                             (CUDA 12.9, PyTorch 2.9.1, Ubuntu 22.04 for MAX compat).
#                             Any NVIDIA CUDA image that boots sshd via PUBLIC_KEY works.
#   --ssh-key-path PATH       Public key to inject. Default: ~/.ssh/id_ed25519.pub
#                             Falls back to ~/.ssh/id_rsa.pub if absent.
#   --capture-to PATH         Default: docs/runpod-bench-<timestamp>.txt
#   --max-boot-min N          Default: 15
#   --max-runtime-hours N     RunPod-side auto-terminate safety net. Default: 2
#                             (belt-and-suspenders in case trap EXIT fails)
#   --dry-run                 Print the launch mutation and exit
#   --help
#
# Example:
#   scripts/runpod-launch.sh --capture-to docs/bench-embed-h100.txt -- \
#     "node packages/embed/bench.js"

set -euo pipefail

# --- config ----------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GPU_TYPE="NVIDIA H100 80GB HBM3"
CLOUD_TYPE="SECURE"
IMAGE="runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204"
SSH_KEY_PATH="$HOME/.ssh/id_ed25519.pub"
BOOTSTRAP="/workspace/persist/bootstrap.sh"
VOLUME_MOUNT="/workspace"
SSH_USER="root"
MAX_BOOT_MIN=15
MAX_RUNTIME_HOURS=2
CAPTURE=""
DRY_RUN=0
COMMAND=""
# Source API key + volume ID from dotfile before reading them into vars.
# (If we read RUNPOD_VOLUME_ID before sourcing, the env-file value never
# reaches the script.)
if [ -f "$HOME/.config/runpod/env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.config/runpod/env"
fi
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set}"

VOLUME_ID="${RUNPOD_VOLUME_ID:-}"
NO_VOLUME=0
# Volume runs keep pixi + weights on /workspace, so 20 GB of container disk is
# plenty. A --no-volume run installs the whole MAX toolchain onto container
# disk instead, which does not fit in 20 GB.
CONTAINER_DISK=20

API="https://api.runpod.io/graphql"

# --- arg parsing -----------------------------------------------------------

usage() { sed -n '2,54p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu-type)      GPU_TYPE="$2";     shift 2 ;;
    --cloud-type)    CLOUD_TYPE="$2";   shift 2 ;;
    --volume-id)     VOLUME_ID="$2";    shift 2 ;;
    --no-volume)     NO_VOLUME=1;       shift   ;;
    --image)         IMAGE="$2";        shift 2 ;;
    --ssh-key-path)  SSH_KEY_PATH="$2"; shift 2 ;;
    --capture-to)    CAPTURE="$2";      shift 2 ;;
    --max-boot-min)  MAX_BOOT_MIN="$2"; shift 2 ;;
    --max-runtime-hours) MAX_RUNTIME_HOURS="$2"; shift 2 ;;
    --dry-run)       DRY_RUN=1;         shift   ;;
    --help|-h)       usage; exit 0 ;;
    --)              shift; COMMAND="$*"; break ;;
    *)               COMMAND="${COMMAND:+$COMMAND }$1"; shift ;;
  esac
done

[ -z "$COMMAND" ] && { echo "error: no command given (use -- 'your command')" >&2; usage >&2; exit 1; }
if [ "$NO_VOLUME" -eq 0 ] && [ -z "$VOLUME_ID" ]; then
  echo "error: --volume-id or RUNPOD_VOLUME_ID required (or pass --no-volume)" >&2
  exit 1
fi

if [ ! -f "$SSH_KEY_PATH" ] && [ -f "$HOME/.ssh/id_rsa.pub" ]; then
  SSH_KEY_PATH="$HOME/.ssh/id_rsa.pub"
fi
[ "$NO_VOLUME" -eq 1 ] && CONTAINER_DISK=80

[ ! -f "$SSH_KEY_PATH" ] && { echo "error: no SSH public key at $SSH_KEY_PATH" >&2; exit 1; }
PUBLIC_KEY=$(cat "$SSH_KEY_PATH")

if [ -z "$CAPTURE" ]; then
  mkdir -p "$REPO_ROOT/docs"
  CAPTURE="$REPO_ROOT/docs/runpod-bench-$(date -u +%Y%m%dT%H%M%SZ).txt"
else
  mkdir -p "$(dirname "$CAPTURE")"
fi

command -v jq >/dev/null || { echo "error: jq not installed" >&2; exit 1; }

# --- GraphQL helper --------------------------------------------------------

# Send a GraphQL query/mutation. $1 = query, $2 = variables (JSON).
gql() {
  local query="$1"; local vars="${2:-{\}}"
  local body
  body=$(jq -nc --arg q "$query" --argjson v "$vars" '{query:$q, variables:$v}')
  curl -fsSL -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -d "$body" "$API"
}

# --- launch ----------------------------------------------------------------

POD_NAME="spike-bench-$(date +%s)"

# Portable ISO8601 ($MAX_RUNTIME_HOURS from now) for terminateAfter safety net.
# RunPod auto-kills the pod at this time even if our trap EXIT fails.
TERM_EPOCH=$(( $(date +%s) + MAX_RUNTIME_HOURS * 3600 ))
if date -u -r 0 >/dev/null 2>&1; then
  TERMINATE_AFTER=$(date -u -r "$TERM_EPOCH" +"%Y-%m-%dT%H:%M:%SZ")   # macOS
else
  TERMINATE_AFTER=$(date -u -d "@$TERM_EPOCH" +"%Y-%m-%dT%H:%M:%SZ") # GNU/Linux
fi

LAUNCH_MUTATION='mutation Launch($input: PodFindAndDeployOnDemandInput) {
  podFindAndDeployOnDemand(input: $input) {
    id
    desiredStatus
  }
}'

LAUNCH_VARS=$(jq -nc \
  --arg name "$POD_NAME" \
  --arg image "$IMAGE" \
  --arg gpu "$GPU_TYPE" \
  --arg cloud "$CLOUD_TYPE" \
  --argjson disk "$CONTAINER_DISK" \
  --arg pubkey "$PUBLIC_KEY" \
  --arg term "$TERMINATE_AFTER" \
  --arg vol "$VOLUME_ID" \
  --arg mount "$VOLUME_MOUNT" \
  --argjson novol "$NO_VOLUME" \
  '{
    input: ({
      name: $name,
      imageName: $image,
      gpuTypeId: $gpu,
      cloudType: $cloud,
      gpuCount: 1,
      containerDiskInGb: $disk,
      volumeInGb: 0,
      ports: "22/tcp",
      startSsh: true,
      supportPublicIp: true,
      terminateAfter: $term,
      env: [{ key: "PUBLIC_KEY", value: $pubkey }]
    } + (if $novol == 1 then {} else {
      networkVolumeId: $vol,
      volumeMountPath: $mount
    } end))
  }')

if [ "$DRY_RUN" = "1" ]; then
  echo "--- dry-run ---"
  echo "mutation:  $LAUNCH_MUTATION"
  echo "variables: $LAUNCH_VARS"
  echo "command:   $COMMAND"
  echo "capture:   $CAPTURE"
  exit 0
fi

# --- trap: always terminate ------------------------------------------------

POD_ID=""
cleanup() {
  local rc=$?
  if [ -n "$POD_ID" ]; then
    echo "[runpod] terminating $POD_ID (exit code $rc)"
    gql 'mutation Term($input: PodTerminateInput!) { podTerminate(input: $input) }' \
        "$(jq -nc --arg id "$POD_ID" '{input:{podId:$id}}')" \
      >/dev/null || echo "[runpod] WARNING: terminate call failed — check dashboard" >&2
  fi
  exit $rc
}
trap cleanup EXIT INT TERM

# --- launch + wait ---------------------------------------------------------

if [ "$NO_VOLUME" -eq 1 ]; then
  echo "[runpod] launching $GPU_TYPE ($CLOUD_TYPE, no volume, ${CONTAINER_DISK}GB disk)"
else
  echo "[runpod] launching $GPU_TYPE ($CLOUD_TYPE, volume=$VOLUME_ID)"
fi
RESP=$(gql "$LAUNCH_MUTATION" "$LAUNCH_VARS")
POD_ID=$(echo "$RESP" | jq -r '.data.podFindAndDeployOnDemand.id // empty')
if [ -z "$POD_ID" ]; then
  echo "[runpod] launch failed: $RESP" >&2
  exit 1
fi
echo "[runpod] pod_id=$POD_ID"

# Poll for runtime.ports to be populated. RunPod reports desiredStatus=RUNNING
# before sshd is actually listening — poll ports + nc test.
POD_QUERY='query Pod($input: PodFilter!) {
  pod(input: $input) {
    id
    desiredStatus
    runtime { ports { ip publicPort privatePort type } }
  }
}'

DEADLINE=$(( $(date +%s) + 60 * MAX_BOOT_MIN ))
IP=""
PORT=""
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  RESP=$(gql "$POD_QUERY" "$(jq -nc --arg id "$POD_ID" '{input:{podId:$id}}')")
  STATUS=$(echo "$RESP" | jq -r '.data.pod.desiredStatus // "unknown"')
  IP=$(echo "$RESP" | jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22 and .type=="tcp") | .ip // empty' | head -1)
  PORT=$(echo "$RESP" | jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22 and .type=="tcp") | .publicPort // empty' | head -1)
  echo "[runpod] status=$STATUS ssh=${IP:-pending}:${PORT:-pending}"
  [ "$STATUS" = "RUNNING" ] && [ -n "$IP" ] && [ -n "$PORT" ] && break
  sleep 15
done

if [ -z "$IP" ] || [ -z "$PORT" ]; then
  echo "[runpod] pod never exposed SSH within ${MAX_BOOT_MIN}m" >&2
  exit 1
fi

echo "[runpod] waiting for ssh at $IP:$PORT"
for _ in $(seq 1 40); do
  if nc -z -w 3 "$IP" "$PORT" 2>/dev/null; then break; fi
  sleep 5
done

# --- run command -----------------------------------------------------------

if [ "$NO_VOLUME" -eq 1 ]; then
# Cold bootstrap: no seeded /workspace/persist to source, so install the
# toolchain onto container disk and clone fresh. Commands mirror section 3 of
# docs/cloud-benchmark-runbook.md. Only valid for a public repo — there are no
# volume-provided credentials here.
PREAMBLE=$(cat <<'EOF'
export DEBIAN_FRONTEND=noninteractive
curl -fsSL https://pixi.sh/install.sh | bash > /tmp/pixi-install.log 2>&1
export PATH="$HOME/.pixi/bin:$PATH"
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash > /tmp/nvm-install.log 2>&1
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm install 22 > /tmp/nvm-node.log 2>&1
nvm use 22 > /dev/null
echo "node $(node --version)  pixi $(pixi --version 2>/dev/null || echo missing)"
cd /workspace 2>/dev/null || cd "$HOME"
if [ ! -d mojo-addon-examples ]; then
  git clone --quiet https://github.com/codetalcott/mojo-addon-examples.git
fi
cd mojo-addon-examples
EOF
)
else
PREAMBLE=$(cat <<EOF
# Soft-source bootstrap: set up PATH + auth, but don't fail the whole run if
# the volume's bootstrap has stale logic (e.g., git-reset pattern). The
# caller's command is expected to do its own repo sync regardless.
source "$BOOTSTRAP" 2>&1 || echo "[warn] bootstrap exited non-zero; continuing"
export PATH="/workspace/persist/bin:/workspace/persist/pixi-home/bin:\$PATH"
export PIXI_CACHE_DIR=/workspace/persist/pixi-cache
export HF_HOME=/workspace/persist/model-cache
cd /workspace/mojo-addon-examples 2>/dev/null || true
EOF
)
fi

REMOTE_SCRIPT=$(cat <<EOF
#!/usr/bin/env bash
set -euo pipefail
$PREAMBLE
echo "=== host \$(hostname) === \$(date -u) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "=== command: $COMMAND ==="
$COMMAND
EOF
)
CMD_B64=$(printf '%s' "$REMOTE_SCRIPT" | base64 | tr -d '\n')

echo "[runpod] running (capture: $CAPTURE)"
{
  echo "# runpod-launch.sh"
  echo "# gpu: $GPU_TYPE  cloud: $CLOUD_TYPE"
  echo "# command: $COMMAND"
  echo "# started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "---"
  ssh -o StrictHostKeyChecking=accept-new \
      -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR \
      -o ServerAliveInterval=30 \
      -p "$PORT" \
      "$SSH_USER@$IP" \
      "echo '$CMD_B64' | base64 -d | bash" 2>&1
  echo "---"
  echo "# finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$CAPTURE"

echo "[runpod] done — output saved to $CAPTURE"
