set -euo pipefail
EXPERIMENT_ROOT=/mnt/nfs/home/mohamadzbib/projects/rl-infra/experiments/dapo-qwen25-math15b-grpo-stale4-3gpu
mkdir -p "$EXPERIMENT_ROOT/validation"
exec 9>"$EXPERIMENT_ROOT/validation/run.lock"
flock -n 9
printf '%s\n' "$SLURM_JOB_ID" > "$EXPERIMENT_ROOT/validation/active-job.txt"
source /mnt/nfs/home/mohamadzbib/projects/rl-infra/env.sh prime-rl
uv run --no-sync python -c 'import importlib.metadata as m,json,os,sys; from pathlib import Path; names=["torch","transformers","vllm","prime-rl","verifiers","pydantic"]; print(json.dumps({"python":sys.version,"packages":{n:m.version(n) for n in names},"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"slurm_job_gpus":os.environ.get("SLURM_JOB_GPUS")}))' > "$EXPERIMENT_ROOT/validation/runtime-$SLURM_JOB_ID.json"
if ! test -s "$EXPERIMENT_ROOT/validation/smoke.json" || ! uv run --no-sync python "$EXPERIMENT_ROOT/scripts/staleness_guard.py" gate --experiment-root "$EXPERIMENT_ROOT"; then
    bash "$EXPERIMENT_ROOT/scripts/smoke_job.sh"
fi
exec bash "$EXPERIMENT_ROOT/scripts/train_job.sh"
