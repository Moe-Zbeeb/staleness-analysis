set -euo pipefail
EXPERIMENT_ROOT=/mnt/nfs/home/mohamadzbib/projects/rl-infra/experiments/dapo-qwen25-math15b-grpo
RL_INFRA=/mnt/nfs/home/mohamadzbib/projects/rl-infra
OBSERVER_ARCHIVE="$RL_INFRA/artifacts/opik-2.2.61-py312.tar"
source "$RL_INFRA/env.sh" prime-rl
BUILD_ROOT=${SLURM_TMPDIR:-/tmp/opik-build-$SLURM_JOB_ID}
OBSERVER_ENV="$BUILD_ROOT/opik-env"
export UV_PROJECT_ENVIRONMENT="$OBSERVER_ENV"
mkdir -p "$BUILD_ROOT" "$(dirname "$OBSERVER_ARCHIVE")"
uv sync --frozen --project "$EXPERIMENT_ROOT/observer" --no-dev
uv run --no-sync --project "$EXPERIMENT_ROOT/observer" python -c 'import opik; assert opik.__version__ == "2.2.61"; print(opik.__version__)'
TEMP_ARCHIVE="$OBSERVER_ARCHIVE.job-$SLURM_JOB_ID.tmp"
tar -C "$BUILD_ROOT" -cf "$TEMP_ARCHIVE" opik-env
VERIFY_ROOT="$BUILD_ROOT/relocation-check"
mkdir -p "$VERIFY_ROOT"
tar -C "$VERIFY_ROOT" -xf "$TEMP_ARCHIVE"
"$VERIFY_ROOT/opik-env/bin/python" -c 'import inspect, opik; assert opik.__version__ == "2.2.61"; assert "feedback_scores" in inspect.signature(opik.Opik.trace).parameters'
mv "$TEMP_ARCHIVE" "$OBSERVER_ARCHIVE"
test -s "$OBSERVER_ARCHIVE"
