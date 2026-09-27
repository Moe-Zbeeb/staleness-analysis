import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util.tensor_util import make_ndarray

from deepseek_study.tracking.readers import JsonlTail
from deepseek_study.tracking.tensorboard import EventSink, observe, scalar_rows


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value) + "\n")


def events(directory):
    rows = []
    for path in sorted(Path(directory).glob("events.out.tfevents.*")):
        for event in EventFileLoader(str(path)).Load():
            for value in event.summary.value:
                if value.HasField("simple_value"):
                    number = value.simple_value
                elif value.HasField("tensor"):
                    array = make_ndarray(value.tensor)
                    if array.size != 1 or array.dtype.kind not in "biuf":
                        continue
                    number = float(array.reshape(-1)[0])
                else:
                    continue
                rows.append((value.tag, event.step, number, event.wall_time))
    return rows


@pytest.fixture
def recorded(tmp_path):
    output = tmp_path / "run"
    write_json(output / "configs/study.json", {"lag": 8, "max_steps": 40, "metrics_mirror_root": None})
    write_json(
        output / "run.json",
        {"run_uuid": "test", "identity_sha256": "source", "config_sha256": "config", "starting_step": 0},
    )
    append(
        output / "metrics.jsonl",
        {
            "producer": "trainer",
            "step": 9,
            "time": 100.25,
            "optim/lr": 0.000001,
            "loss/mean": -0.1,
            "study/noncontributing_token_fraction/mean": 0.37,
        },
    )
    append(output / "metrics.jsonl", {"producer": "trainer", "step": 9, "time": 100.3, "loss/mean": -0.2})
    append(output / "metrics.jsonl", {"producer": "orch", "step": 9, "time": 100.4, "queue/wait_seconds": 0.5})
    append(
        output / "metrics.jsonl",
        {"producer": "orch", "step": None, "time": 100.5, "inference/agg/request_prefill_time_seconds:mean": 1.2},
    )
    append(
        output / "updates.jsonl",
        {
            "step": 9,
            "age_min": 8,
            "age_max": 8,
            "warmup": False,
            "mean_reward": 0.5,
            "queued_versions": list(range(8)),
        },
    )
    append(
        output / "generations.jsonl",
        {
            "policy_version": 0,
            "purpose": "deferred",
            "generation_wall_seconds": 2.5,
            "output_tokens": 100,
            "consumption_step": 9,
        },
    )
    append(
        output / "paper-metrics.jsonl",
        {"step": 9, "metrics": {"clip/fraction": 0.15, "joint/positive/probability_by_ratio/bin_02/mean": 0.75}},
    )
    append(output / "evaluation-metrics.jsonl", {"step": 20, "metrics": {"m2po/aime2024/accuracy": 0.3}})
    write_json(output / "checkpoints/step_20/study/complete.json", {"step": 20})
    return output


def test_all_sources_keep_their_native_metric_clocks(recorded):
    write_json(recorded / "run-status.json", {"status": "killed"})
    original = {path.name: path.read_bytes() for path in recorded.glob("*.jsonl")}
    result = observe(recorded, once=True)
    rows = events(recorded / "tensorboard")
    assert result["status"] == "killed"
    assert result["inference_step_axis"] == "unix_time_milliseconds"
    assert ("trainer/optim/lr", 9, 0.000001, 100.25) in rows
    assert ("trainer/study/noncontributing_token_fraction/mean", 9, 0.37, 100.25) in rows
    assert ("orchestrator/queue/wait_seconds", 9, 0.5, 100.4) in rows
    assert ("inference/agg/request_prefill_time_seconds:mean", 100500, 1.2, 100.5) in rows
    by_tag = {tag: (step, value) for tag, step, value, _ in rows}
    assert by_tag["staleness/age_min"] == (9, 8)
    assert by_tag["staleness/age_max"] == (9, 8)
    assert by_tag["generation/deferred/consumption_step"] == (0, 9)
    assert by_tag["paper/joint/positive/probability_by_ratio/bin_02/mean"] == (9, 0.75)
    assert by_tag["evaluation/m2po/aime2024/accuracy"] == (20, 0.3)
    assert by_tag["checkpoint/completed_step"] == (20, 20)
    assert original == {path.name: path.read_bytes() for path in recorded.glob("*.jsonl")}


def test_restart_deduplicates_completed_events_and_keeps_same_step_observations(recorded):
    write_json(recorded / "run-status.json", {"status": "finished"})
    first = observe(recorded, once=True)
    before = events(recorded / "tensorboard")
    assert first["scalars_written"] == len(before)
    second = observe(recorded, once=True)
    assert second["scalars_written"] == 0
    assert events(recorded / "tensorboard") == before
    assert [value for tag, _, value, _ in before if tag == "trainer/loss/mean"] == [-0.1, -0.2]
    append(recorded / "evaluation-metrics.jsonl", {"step": 40, "metrics": {"m2po/aime2024/accuracy": 0.4}})
    third = observe(recorded, once=True)
    assert third["scalars_written"] == 1
    assert len(events(recorded / "tensorboard")) == len(before) + 1


def test_replay_detects_modified_and_missing_source_observations(tmp_path):
    directory = tmp_path / "events"
    sink = EventSink(directory)
    sink.log({"trainer/loss": 1.0}, 1)
    sink.log({"trainer/loss": 2.0}, 1)
    sink.close()
    sink = EventSink(directory)
    try:
        sink.log({"trainer/loss": 1.0}, 1)
        with pytest.raises(ValueError, match="missing from"):
            sink.validate_replay()
        with pytest.raises(ValueError, match="differs from"):
            sink.log({"trainer/loss": 3.0}, 1)
    finally:
        sink.close()


def test_partial_json_and_utf8_are_not_lost_on_next_poll(tmp_path):
    path = tmp_path / "metrics.jsonl"
    tail = JsonlTail(path)
    assert tail.read() == []
    record = json.dumps({"step": 1, "label": "λ"}, ensure_ascii=False).encode() + b"\n"
    split = record.index("λ".encode()) + 1
    path.write_bytes(record[:split])
    assert tail.read() == []
    with path.open("ab") as stream:
        stream.write(record[split:])
    assert tail.read() == [{"step": 1, "label": "λ"}]
    assert tail.read() == []
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="Append-only"):
        tail.read()


def test_nonfinite_and_private_fields_are_not_scalar_events():
    mapped = scalar_rows(
        "updates.jsonl",
        {"step": 1, "mean_reward": 0.5, "question_ids": ["private"], "invalid": float("nan"), "bad": float("inf")},
    )
    assert mapped == (1, {"rollout/consumed_reward_mean": 0.5}, None)
    assert scalar_rows("metrics.jsonl", {"producer": "orch", "step": None, "inference/queue": 3}) is None


def test_once_refuses_active_run_and_lock_prevents_concurrent_writers(recorded):
    with pytest.raises(ValueError, match="--once"):
        observe(recorded, once=True)
    with (recorded / ".tensorboard-observer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another TensorBoard"):
            observe(recorded)


def test_different_run_cannot_reuse_tensorboard_directory(recorded):
    write_json(recorded / "run-status.json", {"status": "finished"})
    observe(recorded, once=True)
    launch = json.loads((recorded / "run.json").read_text())
    write_json(recorded / "run.json", {**launch, "run_uuid": "other-run"})
    with pytest.raises(ValueError, match="different study run"):
        observe(recorded, once=True)


def test_final_paper_metrics_are_drained_before_observer_exit(recorded):
    write_json(recorded / "paper-observer.json", {"enabled": True})
    process = subprocess.Popen(
        [sys.executable, "-m", "deepseek_study.tracking.tensorboard", str(recorded), "--parent-pid", str(os.getpid())],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while not (recorded / "tracking/tensorboard.json").exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.05)
        write_json(recorded / "run-status.json", {"status": "finished"})
        time.sleep(1.2)
        assert process.poll() is None
        append(recorded / "paper-metrics.jsonl", {"step": 40, "metrics": {"mismatch/m2": 0.031}})
        write_json(recorded / "paper-status.json", {"status": "complete"})
        _, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stderr
        result = json.loads((recorded / "tracking/tensorboard.json").read_text())
        assert result["status"] == "finished"
        assert any(tag == "paper/mismatch/m2" and step == 40 for tag, step, _, _ in events(recorded / "tensorboard"))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_live_stream_does_not_retain_a_second_copy_of_every_scalar(tmp_path):
    sink = EventSink(tmp_path / "events")
    try:
        for step in range(100):
            sink.log({"inference/queue": float(step)}, step)
        assert sink.count == 100
        assert sink.replay is None
        assert sink.replay_directory is None
        assert sink.replay_remaining == 0
    finally:
        sink.close()


def test_transient_mirror_failure_retries_without_losing_event_logging(recorded, monkeypatch, capsys):
    from deepseek_study.tracking.archive import TensorBoardMirror

    write_json(recorded / "run-status.json", {"status": "finished"})
    original = TensorBoardMirror.poll
    attempts = []

    def poll(mirror):
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("temporary archive outage")
        return original(mirror)

    monkeypatch.setattr(TensorBoardMirror, "poll", poll)
    result = observe(recorded, once=True)
    assert result["status"] == "finished"
    assert result["scalars_written"] > 0
    assert len(attempts) >= 2
    assert "will retry" in capsys.readouterr().err


def test_event_writer_close_failure_is_reported_as_observer_failure(recorded, monkeypatch):
    write_json(recorded / "run-status.json", {"status": "finished"})
    original = EventSink.close

    def close(sink):
        original(sink)
        raise OSError("event flush failed")

    monkeypatch.setattr(EventSink, "close", close)
    with pytest.raises(OSError, match="event flush"):
        observe(recorded, once=True)
    result = json.loads((recorded / "tracking/tensorboard.json").read_text())
    assert result["status"] == "crashed"
    assert "failed to close" in result["error"]


def test_restart_index_is_disposable_and_releases_matched_history(tmp_path):
    directory = tmp_path / "events"
    sink = EventSink(directory)
    for step in range(1100):
        sink.log({"trainer/loss": float(step)}, step)
    sink.close()
    sink = EventSink(directory)
    scratch = Path(sink.replay_directory.name)
    try:
        assert sink.replay_remaining == 1100
        assert (scratch / "events.sqlite").is_file()
        for step in range(1100):
            sink.log({"trainer/loss": float(step)}, step)
        sink.validate_replay()
        assert sink.count == 0
        assert sink.replay is None
        assert not scratch.exists()
    finally:
        sink.close()
