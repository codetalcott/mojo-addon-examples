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
#   scripts/runpod-launch.sh --capture-to docs/spike-bench-h100-day5.txt -- \
#     "node spike/bench.js"

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
VOLUME_ID="${RUNPOD_VOLUME_ID:-}"

# Source API key if in dotfile.
if [ -z "${RUNPOD_API_KEY:-}" ] && [ -f "$HOME/.config/runpod/env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.config/runpod/env"
fi
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set}"

API="https://api.runpod.io/graphql"

# --- arg parsing -----------------------------------------------------------

usage() { sed -n '2,54p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu-type)      GPU_TYPE="$2";     shift 2 ;;
    --cloud-type)    CLOUD_TYPE="$2";   shift 2 ;;
    --volume-id)     VOLUME_ID="$2";    shift 2 ;;
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
[ -z "$VOLUME_ID" ] && { echo "error: --volume-id or RUNPOD_VOLUME_ID required" >&2; exit 1; }

if [ ! -f "$SSH_KEY_PATH" ] && [ -f "$HOME/.ssh/id_rsa.pub" ]; then
  SSH_KEY_PATH="$HOME/.ssh/id_rsa.pub"
fi
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
  --arg vol "$VOLUME_ID" \
  --arg mount "$VOLUME_MOUNT" \
  --arg pubkey "$PUBLIC_KEY" \
  --arg term "$TERMINATE_AFTER" \
  '{
    input: {
      name: $name,
      imageName: $image,
      gpuTypeId: $gpu,
      cloudType: $cloud,
      gpuCount: 1,
      containerDiskInGb: 20,
      volumeInGb: 0,
      networkVolumeId: $vol,
      volumeMountPath: $mount,
      ports: "22/tcp",
      startSsh: true,
      supportPublicIp: true,
      terminateAfter: $term,
      env: [{ key: "PUBLIC_KEY", value: $pubkey }]
    }
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

echo "[runpod] launching $GPU_TYPE ($CLOUD_TYPE, volume=$VOLUME_ID)"
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

REMOTE_SCRIPT=$(cat <<EOF
#!/usr/bin/env bash
set -euo pipefail
# Soft-source bootstrap: set up PATH + auth, but don't fail the whole run if
# the volume's bootstrap has stale logic (e.g., git-reset pattern). The
# caller's command is expected to do its own repo sync regardless.
source "$BOOTSTRAP" 2>&1 || echo "[warn] bootstrap exited non-zero; continuing"
# Re-export what bootstrap should have set in case it failed before the exports.
export PATH="/workspace/persist/bin:/workspace/persist/pixi-home/bin:\$PATH"
export PIXI_CACHE_DIR=/workspace/persist/pixi-cache
export HF_HOME=/workspace/persist/model-cache
cd /workspace/mojo-addon-examples 2>/dev/null || true
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
  echo "# pod: $POD_ID  ssh: $IP:$PORT  gpu: $GPU_TYPE  cloud: $CLOUD_TYPE"
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
