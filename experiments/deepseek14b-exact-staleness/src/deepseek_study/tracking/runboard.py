import argparse
import json
import math
import os
import signal
import sys
import threading
import uuid
from pathlib import Path

from deepseek_study import DATASET_ID, MODEL_ID, PRIME_COMMIT


UPDATE_METRICS = {
    "learner_version": "staleness/learner_version",
    "behavior_version": "staleness/behavior_version",
    "age_min": "staleness/age_min",
    "age_max": "staleness/age_max",
    "warmup": "staleness/bootstrap",
    "mean_reward": "rollout/consumed_reward_mean",
    "truncation_fraction": "rollout/consumed_truncation_fraction",
    "zero_advantage_fraction": "rollout/consumed_zero_advantage_fraction",
    "responses": "rollout/consumed_responses",
    "generated_cohorts": "generation/total_cohorts",
    "generated_responses": "generation/total_responses",
    "generated_output_tokens": "generation/total_output_tokens",
    "queue_payload_bytes": "queue/payload_bytes",
    "step_wall_seconds": "study/update_wall_seconds",
}


def warn(message):
    print(f"[study-runboard] {message}", file=sys.stderr, flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def scalar(value):
    if not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def valid_step(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class JsonlTail:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.identity = None

    def read(self):
        try:
            stream = self.path.open("rb")
        except FileNotFoundError:
            return []
        with stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            if self.identity is not None and (identity != self.identity or stat.st_size < self.offset):
                raise ValueError(f"Append-only metric file changed: {self.path.name}")
            self.identity = identity
            stream.seek(self.offset)
            data = stream.read(4 * 1024 * 1024)
        end = data.rfind(b"\n")
        if end < 0:
            if len(data) == 4 * 1024 * 1024:
                raise ValueError(f"Oversized metric record: {self.path.name}")
            return []
        self.offset += end + 1
        records = []
        for line in data[: end + 1].splitlines():
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError):
                warn(f"Skipped malformed record in {self.path.name}")
                continue
            if isinstance(record, dict):
                records.append(record)
        return records


class Metrics:
    def __init__(self, output, run):
        self.output = Path(output)
        self.run = run
        self.tails = {
            name: JsonlTail(self.output / name) for name in ("metrics.jsonl", "updates.jsonl", "generations.jsonl")
        }
        self.checkpoints = {}

    def emit(self, name, record):
        step = record.get("step")
        if name == "metrics.jsonl":
            if record.get("producer") != "trainer":
                return
            values = {
                f"trainer/{key}": value
                for key, value in record.items()
                if key not in {"step", "time", "producer"} and scalar(value)
            }
            if scalar(record.get("time")):
                values["trainer/source_time_unix"] = record["time"]
        elif name == "updates.jsonl":
            values = {metric: record[key] for key, metric in UPDATE_METRICS.items() if scalar(record.get(key))}
            if isinstance(record.get("queued_versions"), list):
                values["queue/cohorts"] = len(record["queued_versions"])
        else:
            step = record.get("policy_version")
            purpose = record.get("purpose")
            if purpose not in {"warmup", "on_policy", "deferred"}:
                return
            values = {
                f"generation/{purpose}/{key}": record[key]
                for key in ("output_tokens", "generation_wall_seconds", "consumption_step")
                if scalar(record.get(key))
            }
        if valid_step(step) and values:
            self.run.log(values, step=step)

    def poll(self):
        advanced = False
        for name, tail in self.tails.items():
            offset = tail.offset
            for record in tail.read():
                self.emit(name, record)
            advanced = advanced or tail.offset != offset
        for marker in sorted((self.output / "checkpoints").glob("step_*/study/complete.json")):
            if str(marker) in self.checkpoints:
                continue
            record = read_json(marker)
            step = record.get("step")
            if not valid_step(step) or marker.parent.parent.name != f"step_{step}":
                raise ValueError("Invalid checkpoint completion step")
            self.run.log({"checkpoint/complete": 1, "checkpoint/completed_step": step}, step=step)
            self.checkpoints[str(marker)] = step
            advanced = True
        return advanced


def terminal_status(output, allow_completion=True):
    path = output / "run-status.json"
    if path.is_file():
        status = read_json(path).get("status")
        if status not in {"finished", "crashed", "killed"}:
            raise ValueError("Invalid launcher termination status")
        return status
    if allow_completion and (output / "study-complete.json").is_file():
        return "finished"
    return None


def create_run(output, run_id):
    from runboard import Run
    from runboard.config import read_server_info

    study = read_json(output / "configs" / "study.json")
    launch = read_json(output / "run.json")
    config = {
        key: value
        for key, value in study.items()
        if key not in {"model_path", "dataset_path", "data_manifest", "prepared_model_path", "output_dir"}
    }
    config.update(
        model_id=MODEL_ID,
        dataset_id=DATASET_ID,
        prime_commit=PRIME_COMMIT,
        source_identity=launch["identity_sha256"],
        config_sha256=launch["config_sha256"],
        starting_step=launch["starting_step"],
        resumed=bool(launch.get("resume_from")),
        intermediate_evaluation=False,
    )
    settings = read_server_info()
    server = os.environ.get("RUNBOARD_SERVER") or settings.get("url")
    token = os.environ.get("RUNBOARD_TOKEN") or settings.get("token")
    options = {}
    if os.environ.get("RUNBOARD_DIR"):
        options = {"mode": "file", "dir": os.environ["RUNBOARD_DIR"]}
    elif not (server and token):
        options = {"mode": "file", "dir": str(output / "tracking" / "runboard-runs")}
    return Run(
        project=os.environ.get("RUNBOARD_PROJECT", "staleness-analysis"),
        name=output.name,
        config=config,
        run_id=run_id,
        tags=["deepseek14b", "exact-staleness", f"lag-{study['lag']}", f"seed-{study['seed']}"],
        **options,
    )


def observe(output, once=False, parent_pid=None, stop=None):
    output = Path(output).resolve()
    stop = stop or threading.Event()
    if once and terminal_status(output) is None:
        raise ValueError("--once requires a completed or launcher-terminated run; omit it to follow live logs")
    run_id = os.environ.get("PRL_RUN_ID") if parent_pid is not None else uuid.uuid4().hex
    run = create_run(output, run_id or uuid.uuid4().hex)
    info = {"project": run.project, "run_id": run.run_id, "mode": run.mode}
    info_path = output / "tracking" / f"runboard-{run.run_id}.json"
    status = "crashed"
    try:
        write_json(info_path, info)
        print(json.dumps(info), flush=True)
        metrics = Metrics(output, run)
        while True:
            advanced = metrics.poll()
            terminal = terminal_status(output, allow_completion=parent_pid is None)
            if stop.is_set():
                terminal = terminal or "killed"
            if parent_pid is not None and os.getppid() != parent_pid:
                terminal = terminal or "crashed"
            if terminal:
                if advanced:
                    continue
                status = terminal
                break
            if not once:
                stop.wait(1)
    except KeyboardInterrupt:
        status = "killed"
        raise
    finally:
        run.finish(status=status, timeout=2)
        write_json(info_path, {**info, "status": status})
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    observe(args.directory, args.once, args.parent_pid, stop)


if __name__ == "__main__":
    main()
