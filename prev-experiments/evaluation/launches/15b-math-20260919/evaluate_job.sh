#!/bin/bash
set -euo pipefail
source /mnt/nfs/home/mohamadzbib/projects/rl-infra/env.sh prime-rl
SWEEP_SOURCE=/mnt/nfs/home/mohamadzbib/projects/rl-infra/evaluation-runs/15b-math-20260919/source
SWEEP_OUTPUT=/mnt/nfs/home/mohamadzbib/projects/rl-infra/outputs/math-sweep-15b-20260919
SWEEP_CACHE=/tmp/math-sweep-15b-$SLURM_JOB_ID
mkdir -p "$SWEEP_CACHE"
export VLLM_CACHE_ROOT="$SWEEP_CACHE/vllm"
export TORCHINDUCTOR_CACHE_DIR="$SWEEP_CACHE/torchinductor"
export TRITON_CACHE_DIR="$SWEEP_CACHE/triton"
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset VLLM_ALLOW_LONG_MAX_MODEL_LEN
cd "$SWEEP_SOURCE"
"$UV_PROJECT_ENVIRONMENT/bin/python" evaluation/launches/15b-math-20260919/dispatch.py run \
  --plan "$SWEEP_OUTPUT/plan.json" \
  --data "$SWEEP_OUTPUT/data.json" \
  --prepared-root "$SWEEP_OUTPUT/prepared" \
  --results-root "$SWEEP_OUTPUT/results" \
  --tokenizer-root /mnt/nfs/home/mohamadzbib/projects/rl-infra \
  --workers-from-visible
exec "$UV_PROJECT_ENVIRONMENT/bin/python" evaluation/sweep.py report \
  --prepared-root "$SWEEP_OUTPUT/prepared" \
  --output-root "$SWEEP_OUTPUT/results" \
  --report "$SWEEP_OUTPUT/report.json"
