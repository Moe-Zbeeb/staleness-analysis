set -euo pipefail
source /mnt/nfs/home/mohamadzbib/projects/rl-infra/env.sh prime-rl
export CUDA_VISIBLE_DEVICES=
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
uv run --no-sync python /mnt/nfs/home/mohamadzbib/projects/rl-infra/recoveries/7b-stale4-normal-retry-20260919/validate_plan.py
