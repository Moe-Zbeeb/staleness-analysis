import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from benchmark_1p5b import run as bounded_command
from launch_full_run import select_devices
from local_backup import atomic_json, digest
from node_local_run import local_environment, run, stage
from run_bounded_profile import records


def validate_case(folder):
    report = json.loads((folder / "timing-summary.json").read_text())
    if report["status"] != "completed" or report["completed_updates"] != 2:
        raise ValueError("Validation must complete exactly two updates")
    updates = records(folder / "run/updates.jsonl")
    if [row["step"] for row in updates] != [1, 2]:
        raise ValueError("Unexpected committed validation updates")
    for row in updates:
        if row["responses"] != 512 or not row["warmup"] or row["age_min"] != 0 or row["age_max"] != 0:
            raise ValueError("Validation changed the bootstrap contract")
        for key in ("step_wall_seconds", "mean_reward", "zero_advantage_fraction"):
            if not math.isfinite(row[key]):
                raise ValueError("Nonfinite validation metric: " + key)
        if row["step_wall_seconds"] <= 0 or row["mean_reward"] <= 0 or row["zero_advantage_fraction"] >= 1:
            raise ValueError("Validation has invalid timing or no learning signal")
    norms = {}
    for row in records(folder / "run/metrics.jsonl"):
        if row.get("producer") == "trainer" and row.get("step") in (1, 2):
            for key, value in row.items():
                if key == "optim/grad_norm" or key.startswith("loss/"):
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError("Nonfinite learner metric: " + key)
            if "optim/grad_norm" in row:
                norms[row["step"]] = row["optim/grad_norm"]
    if set(norms) != {1, 2} or any(value <= 0 for value in norms.values()):
        raise ValueError("Missing or zero learner gradient norms")
    return {"mean_update_seconds": sum(row["step_wall_seconds"] for row in updates) / 2, "report": report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--prepared", action="store_true")
    args = parser.parse_args()
    control = args.control.resolve()
    for name, expected in json.loads((control / "CONTROL_SHA256.json").read_text()).items():
        if digest(control / name) != expected:
            raise ValueError("Launch control changed: " + name)
    spec = json.loads((control / "storage-spec.json").read_text())
    workspace, runtime = Path(spec["workspace"]), Path(spec["runtime"])
    if not args.prepared:
        stage(spec, control)
        python = runtime / "prime-rl/.venv/bin/python"
        environment = local_environment(runtime, workspace, workspace / "release")
        os.execve(python, [str(python), *sys.argv, "--prepared"], environment)
    from deepseek_study.config import StudyConfig

    root = workspace / "validation"
    root.mkdir(exist_ok=False)
    shared = control / ("validation-" + os.environ["SLURM_JOB_ID"])
    shared.mkdir(exist_ok=False)
    environment = dict(os.environ)
    python = sys.executable
    study = StudyConfig.read(workspace / "study.json")
    if (study.trainer_gpus, study.inference_gpus, study.max_steps, study.lag) != (4, 4, 1000, 256):
        raise ValueError("Unexpected study topology or schedule")
    if (study.inference_max_sequences, study.rollout_concurrency, study.trainer_reshard_after_forward) != (
        64,
        256,
        False,
    ):
        raise ValueError("Unexpected production candidate")
    bounded_command(
        [python, str(control / "probe_allocated_gpus.py"), "--output", str(root / "probes.json"), "--timeout", "300"],
        root / "probe.log",
        environment,
        360,
    )
    probes = json.loads((root / "probes.json").read_text())
    devices = select_devices(probes, 8, 8)
    if any("A100" not in row["name"] or row["bytes"] < 79_000_000_000 for row in probes["results"]):
        raise ValueError("Eight healthy A100 80GB GPUs are required")
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    bounded_command(
        [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=8",
            str(workspace / "release/scripts/gpu_health.py"),
            "--expected-gpus",
            "8",
            "--receipt",
            str(root / "health.json"),
        ],
        root / "health.log",
        environment,
        900,
    )
    outcome = {"job_id": os.environ["SLURM_JOB_ID"], "cases": {}, "full_run_authorized": False}
    try:
        for label, sequences, concurrency, reshard in (("baseline", 16, 64, True), ("candidate", 64, 256, False)):
            folder = root / label
            folder.mkdir()
            (folder / "release").symlink_to(workspace / "release", target_is_directory=True)
            values = study.model_dump(mode="json") | {
                "output_dir": str(folder / "run"),
                "metrics_mirror_root": str(folder / "metrics"),
                "inference_max_sequences": sequences,
                "rollout_concurrency": concurrency,
                "trainer_reshard_after_forward": reshard,
            }
            (folder / "study.json").write_text(StudyConfig.model_validate(values).model_dump_json(indent=2))
            bounded_command(
                [
                    python,
                    str(control / "run_bounded_profile.py"),
                    "--directory",
                    str(folder),
                    "--updates",
                    "2",
                    "--seconds",
                    "3600",
                ],
                folder / "supervisor.log",
                environment,
                3750,
            )
            outcome["cases"][label] = validate_case(folder)
            atomic_json(root / "decision.json", outcome)
        baseline = outcome["cases"]["baseline"]["mean_update_seconds"]
        candidate = outcome["cases"]["candidate"]["mean_update_seconds"]
        outcome["measured_bootstrap_speedup"] = baseline / candidate
        outcome["full_run_authorized"] = candidate < baseline
        if not outcome["full_run_authorized"]:
            raise RuntimeError("Candidate did not improve measured update time; full run not started")
    finally:
        atomic_json(root / "decision.json", outcome)
        subprocess.run(
            ["rsync", "-rt", "--exclude=release", str(root) + "/", str(shared) + "/"], check=True, timeout=600
        )
    run(spec)


if __name__ == "__main__":
    main()
