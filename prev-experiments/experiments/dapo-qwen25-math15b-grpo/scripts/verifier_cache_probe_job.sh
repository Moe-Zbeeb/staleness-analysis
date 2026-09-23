set -euo pipefail
RL_INFRA=/mnt/nfs/home/mohamadzbib/projects/rl-infra
PROGRAM="$RL_INFRA/repos/prime-rl/deps/verifiers/verifiers/v1/harnesses/null/program.py"
source "$RL_INFRA/env.sh" prime-rl
NODE_CACHE=${SLURM_TMPDIR:-/tmp/prime-rl-$SLURM_JOB_ID}
VERIFIER_HOME="$NODE_CACHE/verifier-home"
VERIFIER_UV_CACHE="$NODE_CACHE/verifier-uv"
DIGEST=$(sha256sum "$PROGRAM" | awk '{print $1}')
SCRIPT_DIR="$VERIFIER_HOME/.cache/verifiers/runtimes/scripts"
SCRIPT="$SCRIPT_DIR/$DIGEST.py"
mkdir -p "$SCRIPT_DIR" "$VERIFIER_UV_CACHE"
cp "$PROGRAM" "$SCRIPT"
HOME="$VERIFIER_HOME" UV_CACHE_DIR="$VERIFIER_UV_CACHE" uv sync --script "$SCRIPT" -q --no-config
INTERPRETER=$(HOME="$VERIFIER_HOME" UV_CACHE_DIR="$VERIFIER_UV_CACHE" uv python find --script "$SCRIPT" --no-config)
case "$INTERPRETER" in
    "$VERIFIER_UV_CACHE"/*) ;;
    *) exit 1 ;;
esac
"$INTERPRETER" -c 'import httpx, mcp, openai, tenacity'
printf '%s\n' "$SCRIPT" "$INTERPRETER"
