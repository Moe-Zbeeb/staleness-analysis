#!/bin/bash
set -euo pipefail
FULL_RUN_CONTROL=$(cd "${1:?control directory required}" && pwd)
STUDY_ROOT=/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export RUNBOARD_HOME=/mnt/xfs/home/mohamadzbib/.runboard
export RUNBOARD_PROJECT=staleness-analysis
export XDG_CACHE_HOME="$STUDY_ROOT/.cache"
export HF_HOME="$STUDY_ROOT/.cache/huggingface"
export TRITON_CACHE_DIR="$FULL_RUN_CONTROL/triton"
cd "$FULL_RUN_CONTROL"
exec "$STUDY_ROOT/vendor/prime-rl/.venv/bin/python" "$FULL_RUN_CONTROL/prepare_cached_full_run.py" --control "$FULL_RUN_CONTROL"
