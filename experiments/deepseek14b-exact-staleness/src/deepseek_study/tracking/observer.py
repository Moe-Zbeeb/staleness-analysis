import argparse
import fcntl
import json
import os
import signal
import threading
from pathlib import Path

from deepseek_study.config import StudyConfig
from deepseek_study.runtime.checkpoints import atomic_write
from deepseek_study.tracking.archive import MetricMirror
from deepseek_study.tracking.paper import process_step
from deepseek_study.tracking.runboard import JsonlTail, terminal_status, warn


def observe_papers(output, once=False, parent_pid=None, stop=None):
    output = Path(output).resolve()
    with (output / ".paper-observer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another paper metric observer owns this run") from error
        return _observe_papers(output, once, parent_pid, stop)


def _observe_papers(output, once, parent_pid, stop):
    stop = stop or threading.Event()
    study = StudyConfig.read(output / "configs/study.json")
    if once and terminal_status(output) is None:
        raise ValueError("--once requires a terminated training run")
    journal = output / "paper-metrics.jsonl"
    prior = JsonlTail(journal)
    records = []
    while batch := prior.read():
        records.extend(batch)
    if journal.exists() and journal.stat().st_size != prior.offset:
        with journal.open("r+b") as stream:
            stream.truncate(prior.offset)
    done = {record["step"] for record in records}
    clip_sum = sum(record["metrics"]["clip/fraction"] for record in records)
    clipped_count = sum(record["metrics"]["clip/fraction"] * record["metrics"]["tokens/count"] for record in records)
    token_count = sum(record["metrics"]["tokens/count"] for record in records)
    updates = JsonlTail(output / "updates.jsonl")
    pending = {}
    mirror = MetricMirror(output, study.metrics_mirror_root)
    status = "running"
    error = None
    try:
        while True:
            advanced = False
            for receipt in updates.read():
                if receipt["step"] not in done:
                    pending[receipt["step"]] = receipt
            for step in sorted(pending):
                receipt = pending[step]
                record = process_step(output, receipt, study)
                metrics = dict(record["metrics"])
                clip_sum += metrics["clip/fraction"]
                clipped_count += metrics["clip/fraction"] * metrics["tokens/count"]
                token_count += metrics["tokens/count"]
                metrics["clip/run_update_mean"] = clip_sum / (len(done) + 1)
                metrics["clip/run_token_weighted_mean"] = clipped_count / token_count
                with journal.open("a") as stream:
                    stream.write(json.dumps({"step": step, "metrics": metrics}, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                done.add(step)
                advanced = True
            pending.clear()
            try:
                mirror.poll()
                error = None
            except OSError as problem:
                error = f"Metric mirror failed: {type(problem).__name__}"
                warn(error)
            terminal = terminal_status(output, allow_completion=parent_pid is None)
            if stop.is_set() or (parent_pid is not None and os.getppid() != parent_pid):
                terminal = terminal or "killed"
            if terminal and not advanced:
                status = "complete" if error is None else "mirror_failed"
                break
            stop.wait(1)
        result = mirror.finish()
        return {"steps": len(done), "mirror": result}
    except Exception as problem:
        status = "failed"
        error = f"{type(problem).__name__}: {problem}"
        raise
    finally:
        atomic_write(
            output / "paper-status.json", json.dumps({"status": status, "steps": len(done), "error": error}).encode()
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    print(json.dumps(observe_papers(args.directory, args.once, args.parent_pid, stop)))


if __name__ == "__main__":
    main()
