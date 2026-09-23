set -euo pipefail
EXPERIMENT_ROOT=/mnt/nfs/home/mohamadzbib/projects/rl-infra/experiments/dapo-qwen25-math15b-grpo-stale8-5gpu
RL_INFRA=/mnt/nfs/home/mohamadzbib/projects/rl-infra
source "$RL_INFRA/env.sh" prime-rl
test "$(git -C "$RL_INFRA/repos/prime-rl" rev-parse HEAD)" = ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1
test -z "$(git -C "$RL_INFRA/repos/prime-rl" status --porcelain)"
uv run --no-sync python "$EXPERIMENT_ROOT/scripts/staleness_guard.py" materialize --experiment-root "$EXPERIMENT_ROOT"
PYTHONPATH="$EXPERIMENT_ROOT/python" uv run --no-sync python "$EXPERIMENT_ROOT/scripts/validate_experiment.py" --experiment-root "$EXPERIMENT_ROOT"
PYTHONPATH="$EXPERIMENT_ROOT/python" uv run --no-sync rl @ "$EXPERIMENT_ROOT/config/main.toml" --dry-run --clean --output-dir "$EXPERIMENT_ROOT/dry-run" --run.name validated
