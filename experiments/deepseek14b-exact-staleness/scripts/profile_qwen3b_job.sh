#!/bin/bash
set -euo pipefail
PROFILE_CONTROL=$(cd "${1:?control directory required}" && pwd)
STUDY_ROOT=/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study
PROFILE_DIR="$PROFILE_CONTROL/work"
PROFILE_PYTHON="$STUDY_ROOT/vendor/prime-rl/.venv/bin/python"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export RUNBOARD_HOME=/mnt/xfs/home/mohamadzbib/.runboard
export RUNBOARD_PROJECT=staleness-analysis
export XDG_CACHE_HOME="$STUDY_ROOT/.cache"
export HF_HOME="$STUDY_ROOT/.cache/huggingface"
export TRITON_CACHE_DIR="$PROFILE_CONTROL/triton"
export PYTHONPATH="$STUDY_ROOT/releases/correctness-v2-20260926/src"
export PROFILE_ALLOCATED_DEVICES="${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
cd "$PROFILE_CONTROL"
timeout --signal=TERM --kill-after=15s 180s "$PROFILE_PYTHON" "$PROFILE_CONTROL/probe_allocated_gpus.py" \
    --output "$PROFILE_CONTROL/device-probes.json" --timeout 120
PROFILE_GPU_COUNT=$("$PROFILE_PYTHON" - "$PROFILE_CONTROL" <<'PY'
import json
import os
import sys
from pathlib import Path

directory = Path(sys.argv[1])
receipt = json.loads((directory / "device-probes.json").read_text())
allocated = os.environ["PROFILE_ALLOCATED_DEVICES"].split(",")
healthy = receipt["healthy_devices"]
if len(allocated) != 8 or receipt["allocated_devices"] != allocated:
    raise ValueError("Expected the requested eight-GPU node")
if len(healthy) not in {7, 8} or len(set(healthy)) != len(healthy) or not set(healthy).issubset(allocated):
    raise ValueError("Need at least seven verified GPUs")
uuids = [item["uuid"] for item in receipt["results"] if item["healthy"] and item["uuid"] != "unavailable"]
if len(uuids) != len(set(uuids)):
    raise ValueError("Device aliases resolve to duplicate GPUs")
(directory / "participating-devices.txt").write_text(",".join(healthy))
print(len(healthy))
PY
)
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(cat "$PROFILE_CONTROL/participating-devices.txt")
PROFILE_INFERENCE_GPUS=$((PROFILE_GPU_COUNT - 4))
timeout --signal=TERM --kill-after=30s 300s "$PROFILE_PYTHON" -m torch.distributed.run \
    --standalone --nproc-per-node="$PROFILE_GPU_COUNT" \
    "$STUDY_ROOT/releases/correctness-v2-20260926/scripts/gpu_health.py" \
    --expected-gpus "$PROFILE_GPU_COUNT" --receipt "$PROFILE_CONTROL/hardware.json"
"$PROFILE_PYTHON" - "$PROFILE_CONTROL" "$PROFILE_INFERENCE_GPUS" <<'PY'
import json
import os
import sys
from pathlib import Path

directory = Path(sys.argv[1])
inference = int(sys.argv[2])
devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
allocated = os.environ["PROFILE_ALLOCATED_DEVICES"].split(",")
hardware = json.loads((directory / "hardware.json").read_text())
if hardware["world_size"] != len(devices) or len(devices) != inference + 4:
    raise ValueError("Collective validation and process layout disagree")
if not all("A100" in device["name"] and device["bytes"] >= 79_000_000_000 for device in hardware["devices"]):
    raise ValueError("Requested node did not expose A100 80GB GPUs")
record = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "node": os.environ["SLURMD_NODENAME"],
    "allocated_devices": allocated,
    "visible_devices": devices,
    "unused_devices": [device for device in allocated if device not in devices],
    "inference_devices": devices[:inference],
    "trainer_devices": devices[inference:],
    "seven_gpu_fallback_authorized": True,
}
(directory / "allocation.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
PY
"$PROFILE_PYTHON" "$PROFILE_CONTROL/prepare_small_model_profile.py" \
    --source "$STUDY_ROOT/releases/correctness-v2-20260926" \
    --baseline "$STUDY_ROOT/launches/exact256-v2-seed42-20260926T173508Z/study.json" \
    --directory "$PROFILE_DIR" \
    --model Qwen/Qwen2.5-3B --inference-gpus "$PROFILE_INFERENCE_GPUS" \
    --mirror-root /mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics
exec "$PROFILE_PYTHON" "$PROFILE_CONTROL/run_bounded_profile.py" \
    --directory "$PROFILE_DIR" --updates 4 --seconds 18000
