import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def select_checkpoint(previous, config_hash, identity_hash, max_steps):
    candidates = []
    for marker in (previous / "checkpoints").glob("step_*/study/complete.json"):
        directory = marker.parents[1]
        match = re.fullmatch(r"step_(\d+)", directory.name)
        if match is None:
            continue
        record = json.loads(marker.read_text())
        step = int(match.group(1))
        if (
            record.get("format") != 2
            or record.get("step") != step
            or record.get("config_sha256") != config_hash
            or record.get("identity_sha256") != identity_hash
            or not 0 < step <= max_steps
        ):
            raise ValueError(f"Incompatible completed checkpoint: {directory}")
        candidates.append((step, directory))
    if not candidates:
        raise ValueError("No complete checkpoint exists; refusing to restart from initial weights")
    step, checkpoint = max(candidates)
    if step == max_steps:
        raise ValueError("Training checkpoint already reached the budget; inspect finalization instead of retraining")
    return step, checkpoint


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--previous-job", type=int, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    previous, release, control = args.previous.resolve(), args.release.resolve(), args.control.resolve()
    job = os.environ["SLURM_JOB_ID"]
    accounting = subprocess.run(
        ["sacct", "--noheader", "--parsable2", "--jobs", str(args.previous_job), "--format=JobIDRaw,State%30"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    states = [row.split("|", 1)[1].split()[0] for row in accounting if row.startswith(f"{args.previous_job}|")]
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL"}
    if len(states) != 1 or states[0] not in terminal:
        raise RuntimeError(f"Predecessor is not confirmed terminal: {states}")
    workspace = control / f"job-{job}"
    workspace.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(release / "src"))
    from deepseek_study.config import StudyConfig
    from deepseek_study.runtime import checkpoints
    from deepseek_study.runtime.identity import capture, read_identity

    study = StudyConfig.read(previous / "configs/study.json")
    identity = read_identity(previous / "source/identity.json")
    for name, expected in identity["source_files"].items():
        if hashlib.sha256((release / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen predecessor source changed: {name}")
    if capture(release, study)["sha256"] != identity["sha256"]:
        raise ValueError("Runtime or data identity changed since the predecessor")
    complete = previous / "study-complete.json"
    if complete.exists():
        value = json.loads(complete.read_text())
        if value != {"step": study.max_steps, "lag": study.lag, "identity_sha256": identity["sha256"]}:
            raise ValueError("Invalid predecessor completion record")
        write(workspace / "continuation.json", {"status": "not_needed", "previous": str(previous)})
        return
    step, checkpoint = select_checkpoint(previous, study.fingerprint(), identity["sha256"], study.max_steps)
    checkpoints.verify_components(checkpoint)
    output = previous.parent / f"{previous.name}-resume-{job}"
    if output.exists():
        raise FileExistsError("Continuation output already exists")
    resumed = StudyConfig.model_validate({**study.model_dump(), "output_dir": output})
    if resumed.fingerprint() != study.fingerprint():
        raise ValueError("Continuation changed training settings")
    write(workspace / "study.json", resumed.model_dump(mode="json"))
    expected_gpus = study.trainer_gpus + study.inference_gpus
    subprocess.run(
        [sys.executable, str(control / "probe_allocated_gpus.py"), "--output", str(workspace / "device-probes.json")],
        check=True,
        timeout=180,
    )
    probes = json.loads((workspace / "device-probes.json").read_text())
    if len(probes["healthy_devices"]) != expected_gpus or probes["healthy_devices"] != probes["allocated_devices"]:
        raise RuntimeError("Every allocated GPU must pass health and free-memory checks")
    uuids = [item["uuid"] for item in probes["results"] if item["uuid"] != "unavailable"]
    if len(uuids) != len(set(uuids)):
        raise RuntimeError("Allocated device aliases resolve to duplicate GPUs")
    subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={expected_gpus}",
            str(release / "scripts/gpu_health.py"), "--expected-gpus", str(expected_gpus),
            "--receipt", str(workspace / "hardware.json"),
        ],
        check=True,
        timeout=330,
    )
    hardware = json.loads((workspace / "hardware.json").read_text())
    if not all("A100" in item["name"] and item["bytes"] >= 79_000_000_000 for item in hardware["devices"]):
        raise RuntimeError("The continuation requires the original A100 80GB node class")
    command = [
        sys.executable, "-m", "deepseek_study.cli", "run", str(workspace / "study.json"),
        "--resume", str(checkpoint),
    ]
    write(
        workspace / "continuation.json",
        {
            "status": "launching", "previous_job": args.previous_job, "previous": str(previous),
            "checkpoint": str(checkpoint), "starting_step": step, "output": str(output),
            "config_sha256": resumed.fingerprint(), "identity_sha256": identity["sha256"], "command": command,
        },
    )
    os.environ["PYTHONPATH"] = str(release / "src")
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
