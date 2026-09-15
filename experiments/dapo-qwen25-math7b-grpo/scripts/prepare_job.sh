set -euo pipefail
EXPERIMENT_ROOT=/mnt/nfs/home/mohamadzbib/projects/rl-infra/experiments/dapo-qwen25-math7b-grpo
RL_INFRA=/mnt/nfs/home/mohamadzbib/projects/rl-infra
MODEL=/mnt/nfs/home/mohamadzbib/projects/models/Qwen2.5-Math-7B-b101308f
INTELLECT_ROOT="$RL_INFRA/repos/INTELLECT-MATH"
OBSERVER_ARCHIVE="$RL_INFRA/artifacts/opik-2.2.61-py312.tar"
source "$RL_INFRA/env.sh" prime-rl
test "$(git -C "$RL_INFRA/repos/prime-rl" rev-parse HEAD)" = ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1
test "$(git -C "$INTELLECT_ROOT" rev-parse HEAD)" = 0c6d5c96f64bb8981220d8a280a8f998a8b1642f
test -z "$(git -C "$RL_INFRA/repos/prime-rl" status --porcelain)"
test -z "$(git -C "$INTELLECT_ROOT" status --porcelain)"
test -s "$OBSERVER_ARCHIVE"
OBSERVER_VERIFY=${SLURM_TMPDIR:-/tmp/opik-verify-$SLURM_JOB_ID}
mkdir -p "$OBSERVER_VERIFY"
tar -C "$OBSERVER_VERIFY" -xf "$OBSERVER_ARCHIVE"
"$OBSERVER_VERIFY/opik-env/bin/python" -c 'import inspect, opik; assert opik.__version__ == "2.2.61"; assert "feedback_scores" in inspect.signature(opik.Opik.trace).parameters'
uv run --no-sync python "$EXPERIMENT_ROOT/scripts/prepare_data.py" --output-dir "$EXPERIMENT_ROOT/data" --model "$MODEL" --tokenizer-output "$EXPERIMENT_ROOT/tokenizer" --intellect-root "$INTELLECT_ROOT" --exclusions "$EXPERIMENT_ROOT/config/data_exclusions.json" --experiment-root "$EXPERIMENT_ROOT" --observer-archive "$OBSERVER_ARCHIVE" --seed 42
PYTHONPATH="$EXPERIMENT_ROOT/python" uv run --no-sync python "$EXPERIMENT_ROOT/scripts/validate_experiment.py" --experiment-root "$EXPERIMENT_ROOT"
PYTHONPATH="$EXPERIMENT_ROOT/python" uv run --no-sync rl @ "$EXPERIMENT_ROOT/config/main.toml" --dry-run --output-dir "$EXPERIMENT_ROOT/dry-run" --run.name validated
