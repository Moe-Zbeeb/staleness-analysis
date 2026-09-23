set -euo pipefail
EXPERIMENT_ROOT=/mnt/nfs/home/mohamadzbib/projects/rl-infra/experiments/dapo-qwen25-3b-grpo
RL_INFRA=/mnt/nfs/home/mohamadzbib/projects/rl-infra
source "$RL_INFRA/env.sh" prime-rl
export PYTHONPATH="$EXPERIMENT_ROOT/python"
test "$(git -C "$RL_INFRA/repos/prime-rl" rev-parse HEAD)" = ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1
test -z "$(git -C "$RL_INFRA/repos/prime-rl" status --porcelain)"
OUTPUT_DIR=${SLURM_TMPDIR:-/tmp/qwen15b-dryrun-$SLURM_JOB_ID}
exec uv run --no-sync rl @ "$EXPERIMENT_ROOT/config/main.toml" --dry-run --output-dir "$OUTPUT_DIR" --run.name validated-v4
