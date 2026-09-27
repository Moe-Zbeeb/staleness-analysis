import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def records(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.endswith("\n")]


def summarize(output, completed):
    updates = [record for record in records(output / "updates.jsonl") if record["step"] <= completed]
    generations = [
        {key: record.get(key) for key in ("policy_version", "purpose", "output_tokens", "generation_wall_seconds")}
        for record in records(output / "generations.jsonl")
        if record["policy_version"] < completed
    ]
    trainer = []
    path = output / "metrics.jsonl"
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    continue
                record = json.loads(line)
                if record.get("producer") == "trainer" and record.get("step", completed + 1) <= completed:
                    trainer.append(
                        {
                            key: value
                            for key, value in record.items()
                            if key == "step" or key.startswith(("time/", "perf/", "seq_len/", "tokens/"))
                        }
                    )
    return {
        "updates": [
            {
                key: record.get(key)
                for key in (
                    "step",
                    "warmup",
                    "age_min",
                    "age_max",
                    "responses",
                    "step_wall_seconds",
                    "weight_transfer_seconds",
                )
            }
            for record in updates
        ],
        "generations": generations,
        "trainer": trainer,
        "measures_exact256_phase": False,
        "checkpoint_timing_measured": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=18000)
    args = parser.parse_args()
    directory = args.directory.resolve()
    study = json.loads((directory / "study.json").read_text())
    if not 1 <= args.updates < min(study["lag"], study["checkpoint_interval"]):
        raise ValueError("This profiler measures a bounded bootstrap prefix before checkpointing")
    if args.seconds < 1:
        raise ValueError("A positive time bound is required")
    output = Path(study["output_dir"])
    if output.exists():
        raise FileExistsError("Refusing to profile into an existing run")
    environment = {**os.environ, "PYTHONPATH": str(directory / "release/src")}
    process = subprocess.Popen(
        [sys.executable, "-m", "deepseek_study.cli", "run", str(directory / "study.json")],
        env=environment,
    )
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    previous = {number: signal.signal(number, stop) for number in (signal.SIGTERM, signal.SIGINT)}
    started = time.monotonic()
    reason = "launcher_exited"
    completed = 0
    completion_seen = None
    try:
        while process.poll() is None:
            updates = records(output / "updates.jsonl")
            completed = max((row["step"] for row in updates), default=0)
            if completed >= args.updates:
                completion_seen = completion_seen or time.monotonic()
                timing = summarize(output, args.updates)
                if any(
                    row.get("step") == args.updates and "time/forward_backward" in row for row in timing["trainer"]
                ):
                    reason = "requested_updates_completed"
                    break
                if time.monotonic() - completion_seen >= 30:
                    reason = "missing_final_trainer_timing"
                    break
            if interrupted:
                reason = "interrupted"
                break
            if time.monotonic() - started >= args.seconds:
                reason = "profile_timeout"
                break
            time.sleep(0.5)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                reason = "shutdown_timeout"
        for number, handler in previous.items():
            signal.signal(number, handler)
        report = {
            "status": "completed" if reason == "requested_updates_completed" else "incomplete",
            "reason": reason,
            "completed_updates": completed,
            "requested_updates": args.updates,
            "wall_seconds": time.monotonic() - started,
            "launcher_returncode": process.returncode,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
            **summarize(output, completed),
        }
        text = json.dumps(report, indent=2) + "\n"
        (directory / "timing-summary.json").write_text(text)
        mirror = Path(study["metrics_mirror_root"]) / output.name
        if mirror.is_dir():
            (mirror / "timing-summary.json").write_text(text)
        print(text, flush=True)
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
