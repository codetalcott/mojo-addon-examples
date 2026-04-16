#!/usr/bin/env bash
# scripts/lambda-bench.sh — one-shot Lambda Cloud bench runner.
#
# Launches a GPU instance with a persistent filesystem attached, sources the
# repo's bootstrap script (secrets + pixi + gh + git clone), runs a command
# inside mojo-addon-examples/, captures output, and terminates on exit —
# regardless of how the script exits (success, error, Ctrl-C).
#
# Prerequisites (one-time Lambda Cloud setup):
#   - Register your laptop's SSH public key in Lambda dashboard → SSH keys.
#     The name here matches --ssh-key (default: lambda-mojo-bench).
#   - Create a persistent filesystem. Name matches --filesystem
#     (default: mojo-bench-fs).
#   - Seed the filesystem with /home/ubuntu/persist/{secrets/github.env,
#     bootstrap.sh, pixi-home, bin/gh}. See ideas/embedding-kernel-spike-plan.md
#     for the seed commands.
#
# Auth: LAMBDA_API_KEY env var, or ~/.config/lambda/env sourced on start.
#
# Usage:
#   export LAMBDA_API_KEY=...   # or: echo "export LAMBDA_API_KEY=..." > ~/.config/lambda/env
#   scripts/lambda-bench.sh [options] -- "command to run on instance"
#
# Options:
#   --instance-type NAME     Default: gpu_1x_h100_pcie
#   --region NAME            Default: us-east-3
#   --filesystem NAME        Default: mojo-bench-fs
#   --ssh-key NAME           Default: lambda-mojo-bench
#   --capture-to PATH        File path to save combined stdout+stderr
#                            (default: docs/lambda-bench-<timestamp>.txt)
#   --max-boot-min N         How long to wait for instance to become active
#                            (default: 15)
#   --dry-run                Print the launch payload and exit without calling the API
#   --help                   Show this text
#
# Example:
#   scripts/lambda-bench.sh --capture-to docs/spike-bench-h100-day5.txt -- \
#     "node spike/bench.js"

set -euo pipefail

# --- config ----------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INSTANCE_TYPE="gpu_1x_h100_pcie"
REGION="us-east-3"
FILESYSTEM="mojo-bench-fs"
SSH_KEY="lambda-mojo-bench"
SSH_USER="ubuntu"
BOOTSTRAP="/home/ubuntu/persist/bootstrap.sh"
MAX_BOOT_MIN=15
CAPTURE=""
DRY_RUN=0
COMMAND=""

# Source API key from ~/.config/lambda/env if present.
if [ -z "${LAMBDA_API_KEY:-}" ] && [ -f "$HOME/.config/lambda/env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.config/lambda/env"
fi
: "${LAMBDA_API_KEY:?LAMBDA_API_KEY not set (export it, or put it in ~/.config/lambda/env)}"

API="https://cloud.lambda.ai/api/v1"

# --- arg parsing -----------------------------------------------------------

usage() { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
    --region)        REGION="$2";        shift 2 ;;
    --filesystem)    FILESYSTEM="$2";    shift 2 ;;
    --ssh-key)       SSH_KEY="$2";       shift 2 ;;
    --capture-to)    CAPTURE="$2";       shift 2 ;;
    --max-boot-min)  MAX_BOOT_MIN="$2";  shift 2 ;;
    --dry-run)       DRY_RUN=1;          shift   ;;
    --help|-h)       usage; exit 0 ;;
    --)              shift; COMMAND="$*"; break ;;
    *)               COMMAND="${COMMAND:+$COMMAND }$1"; shift ;;
  esac
done

if [ -z "$COMMAND" ]; then
  echo "error: no command given (use -- 'your command')" >&2
  usage >&2
  exit 1
fi

if [ -z "$CAPTURE" ]; then
  mkdir -p "$REPO_ROOT/docs"
  CAPTURE="$REPO_ROOT/docs/lambda-bench-$(date -u +%Y%m%dT%H%M%SZ).txt"
else
  mkdir -p "$(dirname "$CAPTURE")"
fi

command -v jq >/dev/null || { echo "error: jq not installed" >&2; exit 1; }

# --- API helpers -----------------------------------------------------------

# curl wrapper: basic auth (API key as username, empty password), JSON body.
lam() {
  local method="$1"; local api_path="$2"; shift 2
  curl -fsSL -u "$LAMBDA_API_KEY:" -X "$method" \
    -H "Content-Type: application/json" \
    "$API$api_path" "$@"
}

# --- launch payload --------------------------------------------------------

LAUNCH_BODY=$(jq -nc \
  --arg region "$REGION" \
  --arg it "$INSTANCE_TYPE" \
  --arg sshkey "$SSH_KEY" \
  --arg fs "$FILESYSTEM" \
  --arg name "bench-$(date +%s)" \
  '{region_name:$region, instance_type_name:$it, ssh_key_names:[$sshkey],
    file_system_names:[$fs], name:$name, quantity:1}')

if [ "$DRY_RUN" = "1" ]; then
  echo "--- dry-run ---"
  echo "launch payload: $LAUNCH_BODY"
  echo "remote command: $COMMAND"
  echo "capture-to:     $CAPTURE"
  exit 0
fi

# --- trap: always terminate ------------------------------------------------

INSTANCE_ID=""
cleanup() {
  local rc=$?
  if [ -n "$INSTANCE_ID" ]; then
    echo "[lambda] terminating $INSTANCE_ID (exit code $rc)"
    lam POST /instance-operations/terminate \
        -d "$(jq -nc --arg id "$INSTANCE_ID" '{instance_ids:[$id]}')" \
      >/dev/null || echo "[lambda] WARNING: terminate call failed — check dashboard" >&2
  fi
  exit $rc
}
trap cleanup EXIT INT TERM

# --- launch + wait ---------------------------------------------------------

echo "[lambda] launching $INSTANCE_TYPE in $REGION (fs=$FILESYSTEM ssh=$SSH_KEY)"
LAUNCH_RESP=$(lam POST /instance-operations/launch -d "$LAUNCH_BODY")
INSTANCE_ID=$(echo "$LAUNCH_RESP" | jq -r '.data.instance_ids[0] // empty')
if [ -z "$INSTANCE_ID" ]; then
  echo "[lambda] launch failed: $LAUNCH_RESP" >&2
  exit 1
fi
echo "[lambda] instance_id=$INSTANCE_ID"

# Poll for active status. Lambda's instance_operations/launch accepts the
# request but boot can take 2–10 min depending on region and image.
DEADLINE=$(( $(date +%s) + 60 * MAX_BOOT_MIN ))
IP=""
STATUS=""
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  RESP=$(lam GET "/instances/$INSTANCE_ID")
  STATUS=$(echo "$RESP" | jq -r '.data.status // "unknown"')
  IP=$(echo "$RESP" | jq -r '.data.ip // empty')
  echo "[lambda] status=$STATUS ip=${IP:-pending}"
  [ "$STATUS" = "active" ] && [ -n "$IP" ] && break
  sleep 15
done

if [ "$STATUS" != "active" ] || [ -z "$IP" ]; then
  echo "[lambda] instance never became active within ${MAX_BOOT_MIN}m (last status=$STATUS)" >&2
  exit 1
fi

# Wait for SSH. Instance may report active before sshd is listening.
echo "[lambda] waiting for ssh at $IP:22"
for _ in $(seq 1 40); do
  if nc -z -w 3 "$IP" 22 2>/dev/null; then
    break
  fi
  sleep 5
done

# --- run command -----------------------------------------------------------

# Build a remote script and ship it as base64 so quoting in $COMMAND stays
# intact (works on both macOS and Linux — strip base64's line wrapping).
REMOTE_SCRIPT=$(cat <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "$BOOTSTRAP"
cd /home/ubuntu/mojo-addon-examples
echo "=== host \$(hostname) === \$(date -u) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "=== command: $COMMAND ==="
$COMMAND
EOF
)
CMD_B64=$(printf '%s' "$REMOTE_SCRIPT" | base64 | tr -d '\n')

echo "[lambda] running (capture: $CAPTURE)"
{
  echo "# lambda-bench.sh"
  echo "# instance: $INSTANCE_ID  ip: $IP  type: $INSTANCE_TYPE  region: $REGION"
  echo "# command: $COMMAND"
  echo "# started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "---"
  ssh -o StrictHostKeyChecking=accept-new \
      -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR \
      -o ServerAliveInterval=30 \
      "$SSH_USER@$IP" \
      "echo '$CMD_B64' | base64 -d | bash" 2>&1
  echo "---"
  echo "# finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$CAPTURE"

echo "[lambda] done — output saved to $CAPTURE"
# trap terminates the instance on normal exit
