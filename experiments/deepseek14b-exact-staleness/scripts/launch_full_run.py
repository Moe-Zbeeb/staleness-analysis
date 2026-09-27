import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def select_devices(probes, allocated_count, participating_count):
    allocated, healthy = probes["allocated_devices"], probes["healthy_devices"]
    if len(allocated) != allocated_count or len(set(allocated)) != allocated_count:
        raise ValueError("Unexpected Slurm allocation")
    if any("GPU is already occupied:" in row.get("error", "") for row in probes["results"]):
        raise ValueError("An allocated GPU is occupied by another process")
    if len(healthy) != participating_count or len(set(healthy)) != participating_count:
        raise ValueError("GPU health does not match the prepared topology")
    if not set(healthy).issubset(allocated):
        raise ValueError("Probe selected a GPU outside the allocation")
    if allocated_count != participating_count and (allocated_count, participating_count) != (8, 7):
        raise ValueError("Unassigned healthy GPUs are not allowed")
    records = {row["device"]: row for row in probes["results"]}
    if any(not records[device]["healthy"] for device in healthy):
        raise ValueError("Selected GPU failed its probe")
    uuids = [records[device]["uuid"] for device in healthy]
    if "unavailable" in uuids or len(uuids) != len(set(uuids)):
        raise ValueError("Participating GPU identities are unavailable or duplicated")
    return healthy


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    control = args.control.resolve()
    manifest = json.loads((control / "full-run.json").read_text())
    for name, expected in {"study.json": manifest["study_sha256"], **manifest["scripts_sha256"]}.items():
        if hashlib.sha256((control / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Full-run launch artifact changed: {name}")
    if os.environ["SLURMD_NODENAME"] != manifest["node"] or int(os.environ.get("SLURM_RESTART_COUNT", "0")):
        raise ValueError("Unexpected node or unsafe automatic restart")
    release = Path(manifest["release"])
    sys.path.insert(0, str(release / "src"))
    from deepseek_study.config import StudyConfig
    from deepseek_study.runtime.identity import capture

    study = StudyConfig.read(control / "study.json")
    if study.fingerprint() != manifest["config_sha256"]:
        raise ValueError("Full-run configuration changed")
    if capture(release, study)["sha256"] != manifest["identity_sha256"]:
        raise ValueError("Frozen source, runtime or dataset changed")
    if study.output_dir.exists():
        raise FileExistsError("Refusing to overwrite or silently restart an existing run")
    job = control / f"job-{os.environ['SLURM_JOB_ID']}"
    job.mkdir(exist_ok=False)
    allocated = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    subprocess.run(
        [sys.executable, str(control / "probe_allocated_gpus.py"), "--output", str(job / "device-probes.json")],
        check=True, timeout=180,
    )
    probes = json.loads((job / "device-probes.json").read_text())
    if probes["allocated_devices"] != allocated:
        raise ValueError("Probe differs from the current allocation")
    count = study.trainer_gpus + study.inference_gpus
    visible = select_devices(probes, manifest["allocated_gpus"], count)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible)
    os.environ["PYTHONPATH"] = str(release / "src")
    subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={count}",
            str(release / "scripts/gpu_health.py"), "--expected-gpus", str(count),
            "--receipt", str(job / "hardware.json"),
        ],
        check=True, timeout=330,
    )
    hardware = json.loads((job / "hardware.json").read_text())
    if hardware["world_size"] != count or len(hardware["devices"]) != count:
        raise ValueError("Collective world size differs from the run")
    if not all("A100" in row["name"] and row["bytes"] >= manifest["minimum_gpu_bytes"] for row in hardware["devices"]):
        raise ValueError("Allocated GPU hardware differs from the prepared run")
    command = [sys.executable, "-m", "deepseek_study.cli", "run", str(control / "study.json")]
    write(
        job / "launch.json",
        {
            "job_id": os.environ["SLURM_JOB_ID"], "node": manifest["node"],
            "allocated_devices": allocated, "visible_devices": visible,
            "unused_devices": [device for device in allocated if device not in visible],
            "inference_devices": visible[:study.inference_gpus], "trainer_devices": visible[study.inference_gpus:],
            "maximum_updates": study.max_steps, "lag": study.lag, "command": command,
            "config_sha256": manifest["config_sha256"], "identity_sha256": manifest["identity_sha256"],
        },
    )
    os.chdir(release)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
