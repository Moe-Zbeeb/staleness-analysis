#!/bin/bash
set -euo pipefail
source /mnt/nfs/home/mohamadzbib/projects/rl-infra/env.sh prime-rl
SWEEP_SOURCE=/mnt/nfs/home/mohamadzbib/projects/rl-infra/evaluation-runs/15b-math-20260919/source
SWEEP_OUTPUT=/mnt/nfs/home/mohamadzbib/projects/rl-infra/outputs/math-sweep-15b-20260919
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
cd "$SWEEP_SOURCE"
exec "$UV_PROJECT_ENVIRONMENT/bin/python" evaluation/launches/15b-math-20260919/dispatch.py prepare \
  --plan "$SWEEP_OUTPUT/plan.json" \
  --data "$SWEEP_OUTPUT/data.json" \
  --prepared-root "$SWEEP_OUTPUT/prepared" \
  --results-root "$SWEEP_OUTPUT/results" \
  --tokenizer-root /mnt/nfs/home/mohamadzbib/projects/rl-infra
