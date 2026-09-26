import argparse
import asyncio
import json
import os
import runpy
import signal
import subprocess
import sys
from pathlib import Path

from deepseek_study.config import StudyConfig
from deepseek_study.rollouts.audit import audit
from deepseek_study.rollouts.queue import QueueState, run
from deepseek_study.runtime import checkpoints


def write(path, value):
    checkpoints.atomic_write(path, (json.dumps(value, indent=2) + "\n").encode())


def execute(command, directory, label, timeout):
    print(json.dumps({"phase": label, "state": "starting"}), flush=True)
    with (directory / f"{label}.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError(f"Readiness phase {label} exceeded {timeout} seconds") from None
    if code:
        raise RuntimeError(f"Readiness phase {label} failed with status {code}; inspect {label}.log")
    print(json.dumps({"phase": label, "state": "passed"}), flush=True)


async def verify_requested_queue(root, study):
    backend_type = runpy.run_path(str(root / "tests/test_queue.py"))["Backend"]
    backend, state = backend_type(), QueueState(study.lag)
    await run(backend, state, study.max_steps, 4)
    assert [row["age_min"] for row in backend.receipts] == [0] * study.lag + [study.lag] * (
        study.max_steps - study.lag
    )
    assert all(row["age_min"] == row["age_max"] for row in backend.receipts)
    assert state.generated_cohorts == study.max_steps and not state.pending
    return {"lag": study.lag, "updates": study.max_steps, "synthetic_responses_per_cohort": 4}


def inspect_run(output):
    result = audit(output)
    if not result["complete"]:
        raise RuntimeError("Readiness run did not finish")
    paper = [json.loads(line) for line in (output / "paper-metrics.jsonl").read_text().splitlines()]
    if len(paper) != result["updates_in_this_run"]:
        raise RuntimeError("Paper metrics are missing completed updates")
    for row in paper:
        metrics = row["metrics"]
        if not 0 <= metrics["gradient_signal/noncontributing_token_fraction"] <= 1:
            raise RuntimeError("Invalid GRPO contribution fraction")
    status = json.loads((output / "paper-status.json").read_text())
    if status["status"] != "complete":
        raise RuntimeError("Paper observer or metric mirror did not complete")
    result["paper_metric_updates"] = len(paper)
    result["response_tokens"] = sum(row["metrics"]["tokens/count"] for row in paper)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    study = StudyConfig.read(args.study)
    if (study.trainer_gpus, study.inference_gpus) != (4, 4):
        raise ValueError("This readiness test requires the full eight-GPU 80 GB profile")
    if study.output_dir.exists():
        raise FileExistsError("The production output already exists; inspect it before readiness testing")
    os.chdir(root)
    os.environ["RUNBOARD_PROJECT"] = "staleness-analysis-readiness"
    report = {"job_id": os.environ.get("SLURM_JOB_ID"), "status": "running", "production_started": False}
    try:
        execute(
            [sys.executable, "-m", "pytest", "-q", "tests", "--disable-warnings"], directory, "tests", 600
        )
        report["requested_queue"] = asyncio.run(verify_requested_queue(root, study))
        execute(
            [
                sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8",
                str(root / "scripts/gpu_health.py"), "--expected-gpus", "8", "--receipt",
                str(directory / "hardware.json"),
            ],
            directory, "hardware", 600,
        )
        hardware = json.loads((directory / "hardware.json").read_text())
        if any("A100" not in item["name"] or item["bytes"] < 79_000_000_000 for item in hardware["devices"]):
            raise RuntimeError("Readiness requires eight A100 80 GB GPUs")
        report["hardware"] = hardware
        changes = {
            "lag": 1,
            "max_steps": 3,
            "prompts_per_update": 4,
            "lr_warmup_steps": 1,
            "checkpoint_interval": 2,
            "output_dir": directory / f"readiness-{os.environ['SLURM_JOB_ID']}-initial",
        }
        smoke = StudyConfig.model_validate({**study.model_dump(), **changes})
        write(directory / "smoke.json", smoke.model_dump(mode="json"))
        report["diagnostic_overrides"] = smoke.model_dump(mode="json", include=set(changes))
        command = [sys.executable, "-m", "deepseek_study.cli"]
        execute(command + ["run", str(directory / "smoke.json")], directory, "initial", 3600)
        execute(command + ["paper-metrics", str(smoke.output_dir), "--once"], directory, "initial-metrics", 300)
        report["initial"] = inspect_run(smoke.output_dir)
        checkpoint = smoke.output_dir / "checkpoints/step_2"
        checkpoints.verify_components(checkpoint)
        resumed = StudyConfig.model_validate(
            {**smoke.model_dump(), "output_dir": directory / f"readiness-{os.environ['SLURM_JOB_ID']}-resumed"}
        )
        write(directory / "resume.json", resumed.model_dump(mode="json"))
        execute(
            command + ["run", str(directory / "resume.json"), "--resume", str(checkpoint)],
            directory, "resume", 2400,
        )
        execute(command + ["paper-metrics", str(resumed.output_dir), "--once"], directory, "resume-metrics", 300)
        report["resume"] = inspect_run(resumed.output_dir)
        original_rows = [json.loads(line) for line in (smoke.output_dir / "updates.jsonl").read_text().splitlines()]
        resumed_rows = [json.loads(line) for line in (resumed.output_dir / "updates.jsonl").read_text().splitlines()]
        assert len(resumed_rows) == 1 and resumed_rows[0]["step"] == 3
        for key in ("response_ids", "payload_digest", "learner_version", "behavior_version"):
            assert original_rows[-1][key] == resumed_rows[0][key], key
        assert not study.output_dir.exists()
        report.update(status="passed", resumed_original_queued_data=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(directory / "readiness.json", report)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
