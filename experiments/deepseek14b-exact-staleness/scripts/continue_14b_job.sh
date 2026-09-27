#!/bin/bash
set -euo pipefail
CONTINUATION_CONTROL=$(cd "${1:?control directory required}" && pwd)
STUDY_ROOT=/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export DEEPSEEK_STUDY_RUNBOARD=0
export XDG_CACHE_HOME="$STUDY_ROOT/.cache"
export HF_HOME="$STUDY_ROOT/.cache/huggingface"
export TRITON_CACHE_DIR="$STUDY_ROOT/.cache/triton"
export PYTHONPATH="$STUDY_ROOT/releases/correctness-v2-20260926/src"
cd "$STUDY_ROOT/releases/correctness-v2-20260926"
exec "$STUDY_ROOT/vendor/prime-rl/.venv/bin/python" "$CONTINUATION_CONTROL/continue_existing_run.py" \
    --previous "$STUDY_ROOT/outputs/exact256-80gb-seed42-v2" \
    --previous-job 2144963 \
    --release "$STUDY_ROOT/releases/correctness-v2-20260926" \
    --control "$CONTINUATION_CONTROL"
