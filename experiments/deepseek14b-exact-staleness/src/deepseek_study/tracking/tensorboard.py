import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sqlite3
import tempfile
import threading

from deepseek_study.tracking.readers import JsonlTail, read_json, scalar, terminal_status, valid_step, warn, write_json


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
    "training_wait_seconds": "study/training_wait_seconds",
    "weight_transfer_seconds": "study/weight_transfer_seconds",
}
JOURNALS = (
    "metrics.jsonl",
    "updates.jsonl",
    "generations.jsonl",
    "paper-metrics.jsonl",
    "evaluation-metrics.jsonl",
)


def scalar_rows(name, record):
    step = record.get("step")
    walltime = record.get("time") if scalar(record.get("time")) else None
    if name == "metrics.jsonl":
        producer = record.get("producer", "native")
        prefix = {"trainer": "trainer", "orch": "orchestrator", "evals": "native_evaluation"}.get(producer)
        if prefix is None:
            prefix = "native"
        if step is None and walltime is not None:
            step = round(walltime * 1000)
            values = {
                key if key.startswith("inference/") else f"wall_time/{prefix}/{key}": value
                for key, value in record.items()
                if key not in {"step", "time", "producer"} and scalar(value)
            }
        else:
            values = {
                f"{prefix}/{key}": value
                for key, value in record.items()
                if key not in {"step", "time", "producer"} and scalar(value)
            }
    elif name == "updates.jsonl":
        values = {
            UPDATE_METRICS.get(key, f"controller/{key}"): value
            for key, value in record.items()
            if key != "step" and scalar(value)
        }
        if isinstance(record.get("queued_versions"), list):
            values["queue/cohorts"] = len(record["queued_versions"])
    elif name in {"paper-metrics.jsonl", "evaluation-metrics.jsonl"}:
        prefix = "paper" if name == "paper-metrics.jsonl" else "evaluation"
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            return None
        values = {f"{prefix}/{key}": value for key, value in metrics.items() if scalar(value)}
    else:
        step = record.get("policy_version")
        purpose = record.get("purpose")
        if purpose not in {"warmup", "on_policy", "deferred"}:
            return None
        values = {
            f"generation/{purpose}/{key}": value
            for key, value in record.items()
            if key != "policy_version" and scalar(value)
        }
    return (step, values, walltime) if valid_step(step) and values else None


class EventSink:
    def __init__(self, directory):
        from torch.utils.tensorboard import SummaryWriter

        self.replay = None
        self.replay_directory = None
        self.replay_remaining = 0
        self.metadata_present = False
        self.count = 0
        try:
            self._load_existing(directory)
            self.writer = SummaryWriter(log_dir=str(directory), max_queue=100, flush_secs=5)
        except BaseException:
            self._close_replay()
            raise

    def _load_existing(self, directory):
        from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
        from tensorboard.util.tensor_util import make_ndarray

        files = sorted(Path(directory).glob("events.out.tfevents.*"))
        if not files:
            return
        self.replay_directory = tempfile.TemporaryDirectory(prefix="study-tensorboard-replay-")
        self.replay = sqlite3.connect(str(Path(self.replay_directory.name) / "events.sqlite"))
        self.replay.execute("PRAGMA journal_mode=OFF")
        self.replay.execute("PRAGMA synchronous=OFF")
        self.replay.execute("PRAGMA cache_size=-8192")
        self.replay.execute("CREATE TABLE events (ordinal INTEGER PRIMARY KEY, tag TEXT, step INTEGER, value REAL)")
        pending = []
        for path in files:
            for event in EventFileLoader(str(path)).Load():
                for value in event.summary.value:
                    self.metadata_present = self.metadata_present or value.tag == "study/config/text_summary"
                    if value.HasField("simple_value"):
                        number = value.simple_value
                    elif value.HasField("tensor"):
                        array = make_ndarray(value.tensor)
                        if array.size != 1 or array.dtype.kind not in "biuf":
                            continue
                        number = float(array.reshape(-1)[0])
                    else:
                        continue
                    if scalar(number):
                        pending.append((value.tag, event.step, float(number)))
                        self.replay_remaining += 1
                    if len(pending) >= 1000:
                        self.replay.executemany("INSERT INTO events (tag, step, value) VALUES (?, ?, ?)", pending)
                        pending.clear()
        if pending:
            self.replay.executemany("INSERT INTO events (tag, step, value) VALUES (?, ?, ?)", pending)
        self.replay.execute("CREATE INDEX event_key ON events (tag, step, ordinal)")
        self.replay.commit()
        if not self.replay_remaining:
            self._close_replay()

    def _close_replay(self):
        if self.replay is not None:
            self.replay.close()
            self.replay = None
        if self.replay_directory is not None:
            self.replay_directory.cleanup()
            self.replay_directory = None

    def log(self, values, step, walltime=None):
        for tag, value in values.items():
            value = float(value)
            previous = None
            if self.replay is not None:
                previous = self.replay.execute(
                    "SELECT ordinal, value FROM events WHERE tag=? AND step=? ORDER BY ordinal LIMIT 1", (tag, step)
                ).fetchone()
            if previous is not None:
                ordinal, stored_value = previous
                if stored_value != value:
                    raise ValueError(f"TensorBoard scalar differs from its archived source: {tag} at {step}")
                self.replay.execute("DELETE FROM events WHERE ordinal=?", (ordinal,))
                self.replay_remaining -= 1
                if not self.replay_remaining:
                    self._close_replay()
            else:
                self.writer.add_scalar(
                    tag, value, global_step=step, walltime=walltime, new_style=True, double_precision=True
                )
                self.count += 1

    def metadata(self, config):
        if not self.metadata_present:
            self.writer.add_text("study/config", "```json\n" + json.dumps(config, indent=2) + "\n```", global_step=0)

    def validate_replay(self):
        if self.replay_remaining:
            raise ValueError("Archived TensorBoard events are missing from the source journals")

    def flush(self):
        self.writer.flush()

    def close(self):
        try:
            self.writer.close()
        finally:
            self._close_replay()


class Metrics:
    def __init__(self, output, sink):
        self.output = Path(output)
        self.sink = sink
        self.tails = {name: JsonlTail(self.output / name) for name in JOURNALS}
        self.checkpoints = set()

    def poll(self):
        advanced = False
        for name, tail in self.tails.items():
            offset = tail.offset
            for record in tail.read():
                mapped = scalar_rows(name, record)
                if mapped is not None:
                    step, values, walltime = mapped
                    self.sink.log(values, step, walltime)
            advanced = advanced or tail.offset != offset
        for marker in sorted((self.output / "checkpoints").glob("step_*/study/complete.json")):
            if marker in self.checkpoints:
                continue
            step = read_json(marker).get("step")
            if not valid_step(step) or marker.parent.parent.name != f"step_{step}":
                raise ValueError("Invalid checkpoint completion step")
            self.sink.log({"checkpoint/complete": 1, "checkpoint/completed_step": step}, step)
            self.checkpoints.add(marker)
            advanced = True
        return advanced


def _observe(output, once, parent_pid, stop):
    from deepseek_study.tracking.archive import TensorBoardMirror

    if once and terminal_status(output) is None:
        raise ValueError("--once requires a completed or launcher-terminated run")
    study = read_json(output / "configs/study.json")
    launch = read_json(output / "run.json")
    directory = output / "tensorboard"
    owner = {key: launch.get(key) for key in ("run_uuid", "identity_sha256", "config_sha256")}
    marker = directory / "study-owner.json"
    if marker.exists() and read_json(marker) != owner:
        raise ValueError("TensorBoard event directory belongs to a different study run")
    if not marker.exists() and directory.exists() and any(directory.iterdir()):
        raise ValueError("Existing TensorBoard directory has no verified study owner")
    write_json(marker, owner)
    info_path = output / "tracking/tensorboard.json"
    info = {
        "logdir": str(directory),
        "backend": "tensorboard",
        "status": "running",
        "inference_step_axis": "unix_time_milliseconds",
        "generation_step_axis": "behavior_policy_version",
        "training_step_axis": "optimizer_update",
        **owner,
    }
    write_json(info_path, info)
    sink = None
    mirror = None
    status = "crashed"
    error_text = None
    mirror_error = None
    try:
        sink = EventSink(directory)
        sink.metadata(
            {"study": study, "launch": launch, "axes": {key: value for key, value in info.items() if "axis" in key}}
        )
        mirror = TensorBoardMirror(output, study.get("metrics_mirror_root"))
        metrics = Metrics(output, sink)
        print(json.dumps(info), flush=True)
        while True:
            advanced = metrics.poll()
            sink.flush()
            try:
                mirror.poll()
                mirror_error = None
            except OSError as error:
                message = f"TensorBoard mirror will retry: {type(error).__name__}: {error}"
                if message != mirror_error:
                    warn(message)
                mirror_error = message
            terminal = terminal_status(output, allow_completion=parent_pid is None)
            if (
                terminal
                and parent_pid is not None
                and (output / "paper-observer.json").is_file()
                and not (output / "paper-status.json").is_file()
            ):
                terminal = None
            if stop.is_set():
                terminal = terminal or "killed"
            if parent_pid is not None and os.getppid() != parent_pid:
                terminal = terminal or "crashed"
            if terminal and not advanced:
                sink.validate_replay()
                status = terminal
                break
            if not terminal:
                stop.wait(1)
    except BaseException as error:
        error_text = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if sink is not None:
                sink.close()
        except BaseException as error:
            status = "crashed"
            error_text = f"TensorBoard writer failed to close: {type(error).__name__}: {error}"
            raise
        finally:
            result = {**info, "status": status, "scalars_written": sink.count if sink else 0, "error": error_text}
            write_json(info_path, result)
            if mirror is not None:
                try:
                    mirror.finish()
                except Exception as error:
                    result.update(
                        status="crashed", error=f"TensorBoard mirror failed: {type(error).__name__}: {error}"
                    )
                    write_json(info_path, result)
                    warn(result["error"])
                    if error_text is None:
                        raise
    return result


def observe(output, once=False, parent_pid=None, stop=None):
    output = Path(output).resolve()
    stop = stop or threading.Event()
    with (output / ".tensorboard-observer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another TensorBoard observer owns this run") from error
        return _observe(output, once, parent_pid, stop)


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
