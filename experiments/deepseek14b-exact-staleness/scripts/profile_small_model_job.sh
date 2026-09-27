#!/bin/bash
set -euo pipefail
PROFILE_CONTROL=$(cd "${1:?control directory required}" && pwd)
STUDY_ROOT=/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study
PROFILE_DIR="$PROFILE_CONTROL/work"
PROFILE_PYTHON="$STUDY_ROOT/vendor/prime-rl/.venv/bin/python"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export RUNBOARD_HOME=/mnt/xfs/home/mohamadzbib/.runboard
export RUNBOARD_PROJECT=staleness-analysis
export XDG_CACHE_HOME="$STUDY_ROOT/.cache"
export HF_HOME="$STUDY_ROOT/.cache/huggingface"
export TRITON_CACHE_DIR="$PROFILE_CONTROL/triton"
export PYTHONPATH="$STUDY_ROOT/releases/correctness-v2-20260926/src"
cd "$PROFILE_CONTROL"
timeout --signal=TERM --kill-after=30s 300s "$PROFILE_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 \
    "$STUDY_ROOT/releases/correctness-v2-20260926/scripts/gpu_health.py" \
    --expected-gpus 8 --receipt "$PROFILE_CONTROL/hardware.json"
"$PROFILE_PYTHON" - "$PROFILE_CONTROL" <<'PY'
import json
import os
import sys
from pathlib import Path

directory = Path(sys.argv[1])
hardware = json.loads((directory / "hardware.json").read_text())
if hardware["world_size"] != 8 or len(hardware["devices"]) != 8:
    raise ValueError("Expected eight GPU ranks")
if not all("A100" in device["name"] and device["bytes"] >= 39_000_000_000 for device in hardware["devices"]):
    raise ValueError("The profile requires eight A100 GPUs with at least 40GB nominal memory")
visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
if len(visible) != len(set(visible)) or len(visible) != 8:
    raise ValueError("Invalid full-node GPU allocation")
record = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "node": os.environ["SLURMD_NODENAME"],
    "visible_devices": visible,
    "inference_devices": visible[:4],
    "trainer_devices": visible[4:],
}
(directory / "allocation.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
PY
nvidia-smi topo -m > "$PROFILE_CONTROL/topology.txt"
"$PROFILE_PYTHON" "$PROFILE_CONTROL/prepare_small_model_profile.py" \
    --source "$STUDY_ROOT/releases/correctness-v2-20260926" \
    --baseline "$STUDY_ROOT/launches/exact256-v2-seed42-20260926T173508Z/study.json" \
    --directory "$PROFILE_DIR" \
    --mirror-root /mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics
exec "$PROFILE_PYTHON" "$PROFILE_CONTROL/run_bounded_profile.py" \
    --directory "$PROFILE_DIR" --updates 4 --seconds 18000
