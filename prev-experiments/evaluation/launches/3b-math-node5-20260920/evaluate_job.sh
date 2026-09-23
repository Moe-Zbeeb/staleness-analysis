#!/bin/bash
set -euo pipefail
export RECOVERY_SLURM_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
source /mnt/nfs/home/mohamadzbib/projects/rl-infra/env.sh prime-rl
SWEEP_PROJECT=/mnt/nfs/home/mohamadzbib/projects/rl-infra
SWEEP_SOURCE="$SWEEP_PROJECT/evaluation-runs/3b-math-node5-20260920/source"
SWEEP_OUTPUT="$SWEEP_PROJECT/outputs/math-sweep-3b-node5-20260920"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd "$SWEEP_SOURCE"
SWEEP_CACHE=/tmp/math-sweep-3b-$SLURM_JOB_ID
mkdir -p "$SWEEP_CACHE"
export VLLM_CACHE_ROOT="$SWEEP_CACHE/vllm"
export TORCHINDUCTOR_CACHE_DIR="$SWEEP_CACHE/torchinductor"
export TRITON_CACHE_DIR="$SWEEP_CACHE/triton"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset VLLM_ALLOW_LONG_MAX_MODEL_LEN
exec "$UV_PROJECT_ENVIRONMENT/bin/python" evaluation/launches/3b-math-node5-20260920/dispatch.py run --workers-from-visible \
  --plan "$SWEEP_OUTPUT/plan.json" \
  --data "$SWEEP_OUTPUT/data.json" \
  --prepared-root "$SWEEP_OUTPUT/prepared" \
  --results-root "$SWEEP_OUTPUT/results" \
  --tokenizer-root "$SWEEP_PROJECT"
