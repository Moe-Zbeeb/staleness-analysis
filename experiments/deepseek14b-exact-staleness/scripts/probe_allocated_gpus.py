import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


PROBE = """
import faulthandler
import json
import os
import sys
import time

started = time.monotonic()

def stage(name):
    print('[gpu-probe-stage]' + json.dumps({'stage': name, 'seconds': time.monotonic() - started}), file=sys.stderr, flush=True)

trace_after = int(os.environ.get('DEEPSEEK_STUDY_GPU_PROBE_TRACE_AFTER', '0'))
if trace_after > 0:
    faulthandler.dump_traceback_later(trace_after, repeat=True)
stage('before_torch_import')
import torch
stage('after_torch_import')

stage('before_device_count')
count = torch.cuda.device_count()
stage('after_device_count')
if count != 1:
    raise RuntimeError(f"Expected one visible CUDA device, got {count}")
stage('before_set_device')
torch.cuda.set_device(0)
stage('after_set_device')
stage('before_mem_info')
free_bytes, total_bytes = torch.cuda.mem_get_info()
stage('after_mem_info')
if free_bytes < 0.9 * total_bytes:
    raise RuntimeError(f"GPU is already occupied: {free_bytes} of {total_bytes} bytes free; need at least 90% free before profiling")
stage('before_bf16_allocation')
layer = torch.nn.Linear(128, 128, dtype=torch.bfloat16, device="cuda")
value = torch.randn(32, 128, dtype=torch.bfloat16, device="cuda")
stage('before_bf16_backward')
layer(value).float().square().mean().backward()
stage('before_cuda_synchronize')
torch.cuda.synchronize()
stage('after_cuda_synchronize')
if not torch.isfinite(layer.weight.grad).all().item():
    raise RuntimeError("Nonfinite BF16 gradient")
stage('before_device_properties')
device = torch.cuda.get_device_properties(0)
stage('complete')
if trace_after > 0:
    faulthandler.cancel_dump_traceback_later()
print(json.dumps({"name": device.name, "bytes": device.total_memory, "uuid": str(getattr(device, "uuid", "unavailable")), "initial_free_bytes": free_bytes, "initial_total_bytes": total_bytes, "bf16_backward": True}))
"""


def probe_diagnostics(stdout, stderr):
    def decode(value):
        return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

    output, errors = decode(stdout), decode(stderr)
    stages = []
    for line in errors.splitlines():
        if not line.startswith("[gpu-probe-stage]"):
            continue
        try:
            record = json.loads(line.removeprefix("[gpu-probe-stage]"))
        except ValueError:
            continue
        if isinstance(record, dict) and isinstance(record.get("stage"), str):
            stages.append(record)
    return {
        "stdout_tail": output[-4000:],
        "stderr_tail": errors[-8000:],
        "last_stage": stages[-1]["stage"] if stages else None,
        "stages": stages[-32:],
    }


def probe_device(device, timeout=120, trace_after_seconds=0):
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": device,
        "OMP_NUM_THREADS": "1",
        "DEEPSEEK_STUDY_GPU_PROBE_TRACE_AFTER": str(trace_after_seconds),
    }
    try:
        result = subprocess.run(
            [sys.executable, "-c", PROBE], env=environment, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as error:
        return {
            "device": device,
            "healthy": False,
            "error": "probe_timeout",
            **probe_diagnostics(error.stdout, error.stderr),
        }
    diagnostics = probe_diagnostics(result.stdout, result.stderr)
    if result.returncode:
        return {"device": device, "healthy": False, "error": result.stderr[-4000:], **diagnostics}
    return {"device": device, "healthy": True, **json.loads(result.stdout.splitlines()[-1]), **diagnostics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--trace-after-seconds", type=int, default=0)
    args = parser.parse_args()
    if args.timeout < 1 or args.trace_after_seconds < 0:
        parser.error("Timeout must be positive and traceback delay must be nonnegative")
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices) != len(set(devices)) or not all(devices):
        raise ValueError("Invalid Slurm device list")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        results = list(pool.map(lambda device: probe_device(device, args.timeout, args.trace_after_seconds), devices))
    receipt = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "allocated_devices": devices,
        "healthy_devices": [record["device"] for record in results if record["healthy"]],
        "results": results,
        "collective_health_verified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)
    occupied = [record["device"] for record in results if "GPU is already occupied:" in record.get("error", "")]
    if occupied:
        raise SystemExit(f"Allocated GPUs are occupied by existing work: {occupied}; refusing to start profiling")


if __name__ == "__main__":
    main()
