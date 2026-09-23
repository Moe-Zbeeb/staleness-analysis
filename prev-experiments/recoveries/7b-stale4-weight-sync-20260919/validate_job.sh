set -euo pipefail
source /mnt/nfs/home/mohamadzbib/projects/rl-infra/env.sh prime-rl
RECOVERY_ROOT=/mnt/nfs/home/mohamadzbib/projects/rl-infra/recoveries/7b-stale4-weight-sync-20260919
EXPERIMENT_ROOT=/mnt/nfs/home/mohamadzbib/projects/rl-infra/experiments/dapo-qwen25-math7b-grpo-stale4-5gpu
export PYTHONPATH="$RECOVERY_ROOT:$RECOVERY_ROOT/prime-rl-source/src:$RECOVERY_ROOT/prime-rl-source/packages/prime-rl-configs/src:$EXPERIMENT_ROOT/python"
export CUDA_VISIBLE_DEVICES=
uv run --no-sync python "$RECOVERY_ROOT/runtime_guard.py" --upstream-root /mnt/nfs/home/mohamadzbib/projects/rl-infra/repos/prime-rl --output "$RECOVERY_ROOT/validation/runtime-preflight-job-$SLURM_JOB_ID.json"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
uv run --no-sync python /mnt/nfs/home/mohamadzbib/projects/rl-infra/recoveries/7b-stale4-weight-sync-20260919/validate_plan.py
