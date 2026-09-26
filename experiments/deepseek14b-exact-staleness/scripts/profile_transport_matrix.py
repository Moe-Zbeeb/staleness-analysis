import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpus", type=int, required=True)
    parser.add_argument("--group-sizes", type=int, nargs="+", required=True)
    args = parser.parse_args()
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(set(devices)) != len(devices) or len(devices) != args.expected_gpus:
        raise ValueError("Visible GPUs differ from the full allocation")
    if sum(args.group_sizes) != len(devices) or min(args.group_sizes) < 2:
        raise ValueError("Every GPU must belong to exactly one collective group")
    args.output.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    for key in ("NCCL_PROTO", "NCCL_MIN_CTAS", "NCCL_MAX_CTAS", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS"):
        environment.pop(key, None)
    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={len(devices)}"]

    def run(name, command, env, timeout=600):
        print(json.dumps({"starting": name}), flush=True)
        with (args.output / f"{name}.log").open("x") as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=True)

    run("topology", ["nvidia-smi", "topo", "-m"], environment, timeout=60)
    run(
        "health",
        launcher
        + [
            str(scripts / "gpu_health.py"),
            "--expected-gpus",
            str(len(devices)),
            "--receipt",
            str(args.output / "health.json"),
        ],
        environment,
    )
    candidates = [
        ("network-auto", "network", {}),
        ("network-simple", "network", {"NCCL_PROTO": "Simple"}),
        ("peer-auto", "peer", {}),
        ("peer-simple", "peer", {"NCCL_PROTO": "Simple"}),
        ("peer-simple-8ctas", "peer", {"NCCL_PROTO": "Simple", "NCCL_MIN_CTAS": "8", "NCCL_MAX_CTAS": "8"}),
    ]
    manifest = {
        "job_id": os.environ["SLURM_JOB_ID"],
        "node": os.environ["SLURMD_NODENAME"],
        "visible_devices": devices,
        "groups": args.group_sizes,
        "candidates": candidates,
        "scripts_sha256": {
            name: hashlib.sha256((scripts / name).read_bytes()).hexdigest()
            for name in ("gpu_health.py", "profile_collectives.py", "profile_transport_matrix.py")
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, transport, overrides in candidates:
        value = "1" if transport == "network" else "0"
        env = {**environment, "NCCL_P2P_DISABLE": value, "NCCL_SHM_DISABLE": value, "NCCL_DEBUG": "INFO", **overrides}
        command = launcher + [
            str(scripts / "profile_collectives.py"),
            "--transport",
            transport,
            "--output",
            str(args.output / name),
            "--repeats",
            "6",
            "--sizes-mib",
            "64",
            "512",
            "1024",
            "--group-sizes",
            *map(str, args.group_sizes),
        ]
        run(name, command, env)
    print(json.dumps({"status": "completed", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
