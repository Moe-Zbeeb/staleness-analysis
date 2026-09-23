#!/bin/bash
set -euo pipefail
source /mnt/nfs/home/mohamadzbib/projects/rl-infra/env.sh prime-rl
SWEEP_PROJECT=/mnt/nfs/home/mohamadzbib/projects/rl-infra
SWEEP_SOURCE="$SWEEP_PROJECT/evaluation-runs/15b-math-20260919/source"
SWEEP_LAUNCH="$SWEEP_PROJECT/evaluation-runs/15b-math-normal-v2-20260919/source"
SWEEP_GUARD="$SWEEP_PROJECT/evaluation-runs/15b-math-normal-v2-occupancy-20260919/source"
SWEEP_ORIGINAL="$SWEEP_PROJECT/outputs/math-sweep-15b-20260919"
SWEEP_OUTPUT="$SWEEP_PROJECT/outputs/math-sweep-15b-normal-v2-20260919"
SWEEP_CACHE=/tmp/math-sweep-15b-normal-$SLURM_JOB_ID
mkdir -p "$SWEEP_CACHE"
export VLLM_CACHE_ROOT="$SWEEP_CACHE/vllm"
export TORCHINDUCTOR_CACHE_DIR="$SWEEP_CACHE/torchinductor"
export TRITON_CACHE_DIR="$SWEEP_CACHE/triton"
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset VLLM_ALLOW_LONG_MAX_MODEL_LEN
cd "$SWEEP_SOURCE"
exec "$UV_PROJECT_ENVIRONMENT/bin/python" "$SWEEP_GUARD/resume.py" \
  --frozen-launcher "$SWEEP_LAUNCH" \
  --source-root "$SWEEP_SOURCE" \
  --plan "$SWEEP_ORIGINAL/plan.json" \
  --data "$SWEEP_ORIGINAL/data.json" \
  --prepared-root "$SWEEP_ORIGINAL/prepared" \
  --derived-prepared-root "$SWEEP_OUTPUT/prepared" \
  --results-root "$SWEEP_OUTPUT/results" \
  --tokenizer-root "$SWEEP_PROJECT" \
  --gres-config /usr/local/etc/gres.conf \
  --expected-gres-sha256 1a5c2624ff52be8400be632e1aed6f61b75399446b91d9eb249d19ac38307732
