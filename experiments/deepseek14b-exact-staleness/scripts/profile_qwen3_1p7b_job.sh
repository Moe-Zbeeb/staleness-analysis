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
"$PROFILE_PYTHON" - "$PROFILE_CONTROL" <<'PY'
import json
import os
import sys
from pathlib import Path

directory = Path(sys.argv[1])
receipt = json.loads((directory / "device-probes.json").read_text())
allocated = os.environ["PROFILE_ALLOCATED_DEVICES"].split(",")
if len(allocated) != 9 or len(set(allocated)) != 9 or receipt["allocated_devices"] != allocated:
    raise ValueError("Expected the requested full nine-GPU allocation")
if receipt["healthy_devices"] != allocated:
    raise ValueError("All nine allocated GPUs must pass before model preparation")
uuids = [item["uuid"] for item in receipt["results"] if item["uuid"] != "unavailable"]
if len(uuids) != len(set(uuids)):
    raise ValueError("Device aliases resolve to duplicate GPUs")
PY
timeout --signal=TERM --kill-after=30s 300s "$PROFILE_PYTHON" -m torch.distributed.run \
    --standalone --nproc-per-node=9 \
    "$STUDY_ROOT/releases/correctness-v2-20260926/scripts/gpu_health.py" \
    --expected-gpus 9 --receipt "$PROFILE_CONTROL/hardware.json"
"$PROFILE_PYTHON" - "$PROFILE_CONTROL" <<'PY'
import json
import os
import sys
from pathlib import Path

directory = Path(sys.argv[1])
devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
allocated = os.environ["PROFILE_ALLOCATED_DEVICES"].split(",")
hardware = json.loads((directory / "hardware.json").read_text())
if hardware["world_size"] != 9 or len(hardware["devices"]) != 9 or devices != allocated:
    raise ValueError("Collective validation and full-node process layout disagree")
if not all("A100" in device["name"] and device["bytes"] >= 39_000_000_000 for device in hardware["devices"]):
    raise ValueError("The profile requires A100 GPUs with at least 40GB nominal memory")
record = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "node": os.environ["SLURMD_NODENAME"],
    "allocated_devices": allocated,
    "visible_devices": devices,
    "unused_devices": [],
    "inference_devices": devices[:5],
    "trainer_devices": devices[5:],
}
(directory / "allocation.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
PY
nvidia-smi topo -m > "$PROFILE_CONTROL/topology.txt"
"$PROFILE_PYTHON" "$PROFILE_CONTROL/prepare_small_model_profile.py" \
    --source "$STUDY_ROOT/releases/correctness-v2-20260926" \
    --baseline "$STUDY_ROOT/launches/exact256-v2-seed42-20260926T173508Z/study.json" \
    --directory "$PROFILE_DIR" \
    --model Qwen/Qwen3-1.7B --inference-gpus 5 \
    --mirror-root /mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics
exec "$PROFILE_PYTHON" "$PROFILE_CONTROL/run_bounded_profile.py" \
    --directory "$PROFILE_DIR" --updates 4 --seconds 18000
