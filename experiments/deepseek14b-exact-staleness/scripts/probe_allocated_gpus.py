import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


PROBE = """
import json
import torch

if torch.cuda.device_count() != 1:
    raise RuntimeError(f"Expected one visible CUDA device, got {torch.cuda.device_count()}")
torch.cuda.set_device(0)
layer = torch.nn.Linear(128, 128, dtype=torch.bfloat16, device="cuda")
value = torch.randn(32, 128, dtype=torch.bfloat16, device="cuda")
layer(value).float().square().mean().backward()
torch.cuda.synchronize()
if not torch.isfinite(layer.weight.grad).all().item():
    raise RuntimeError("Nonfinite BF16 gradient")
device = torch.cuda.get_device_properties(0)
print(json.dumps({"name": device.name, "bytes": device.total_memory, "uuid": str(getattr(device, "uuid", "unavailable")), "bf16_backward": True}))
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices) != len(set(devices)) or not all(devices):
        raise ValueError("Invalid Slurm device list")

    def probe(device):
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": device, "OMP_NUM_THREADS": "1"}
        try:
            result = subprocess.run(
                [sys.executable, "-c", PROBE], env=environment, capture_output=True, text=True, timeout=args.timeout
            )
        except subprocess.TimeoutExpired:
            return {"device": device, "healthy": False, "error": "probe_timeout"}
        if result.returncode:
            return {"device": device, "healthy": False, "error": result.stderr[-4000:]}
        return {"device": device, "healthy": True, **json.loads(result.stdout.splitlines()[-1])}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        results = list(pool.map(probe, devices))
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


if __name__ == "__main__":
    main()
