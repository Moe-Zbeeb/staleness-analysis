#!/bin/bash
set -euo pipefail
source /mnt/nfs/home/mohamadzbib/projects/rl-infra/env.sh prime-rl
SWEEP_PROJECT=/mnt/nfs/home/mohamadzbib/projects/rl-infra
SWEEP_SOURCE="$SWEEP_PROJECT/evaluation-runs/3b-math-node5-20260920/source"
SWEEP_OUTPUT="$SWEEP_PROJECT/outputs/math-sweep-3b-node5-20260920"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd "$SWEEP_SOURCE"
exec "$UV_PROJECT_ENVIRONMENT/bin/python" evaluation/launches/3b-math-node5-20260920/report.py \
  --plan "$SWEEP_OUTPUT/plan.json" \
  --data "$SWEEP_OUTPUT/data.json" \
  --prepared-root "$SWEEP_OUTPUT/prepared" \
  --results-root "$SWEEP_OUTPUT/results" \
  --tokenizer-root "$SWEEP_PROJECT"
