import json
import os
import subprocess
import sys
import time
import urllib.request
from types import SimpleNamespace

import pytest
from runboard.server import Server
from runboard.storage import Storage

from deepseek_study.recipe import baseline
from deepseek_study.tracking.runboard import JsonlTail, Metrics, observe


@pytest.fixture(autouse=True)
def isolated_runboard(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNBOARD_HOME", str(tmp_path / "runboard-home"))
    for key in (
        "RUNBOARD_SERVER",
        "RUNBOARD_TOKEN",
        "RUNBOARD_DIR",
        "RUNBOARD_PROJECT",
        "PRL_RUN_ID",
        "RANK",
        "LOCAL_RANK",
        "DEEPSEEK_STUDY_RUNBOARD",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(key, raising=False)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def append(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


@pytest.fixture
def recorded(tmp_path):
    output = tmp_path / "study-output"
    study = baseline(32).model_copy(update={"output_dir": output})
    write_json(output / "configs/study.json", study.model_dump(mode="json"))
    write_json(
        output / "run.json",
        {
            "identity_sha256": "source",
            "config_sha256": study.fingerprint(),
            "starting_step": 0,
            "resume_from": None,
        },
    )
    append(output / "metrics.jsonl", {"producer": "trainer", "step": 33, "time": 123, "loss/mean": -0.1})
    append(output / "metrics.jsonl", {"producer": "orch", "step": 33, "unwanted": 99})
    append(
        output / "updates.jsonl",
        {
            "step": 33,
            "learner_version": 32,
            "behavior_version": 0,
            "age_min": 32,
            "age_max": 32,
            "warmup": False,
            "mean_reward": 0.5,
            "responses": 512,
            "queued_versions": list(range(1, 33)),
            "response_ids": ["private-response"],
            "question_ids": ["private-question"],
        },
    )
    append(
        output / "generations.jsonl",
        {
            "policy_version": 0,
            "purpose": "deferred",
            "consumption_step": 33,
            "output_tokens": 1024,
            "generation_wall_seconds": 2.5,
            "response_ids": ["private-response"],
        },
    )
    write_json(output / "checkpoints/step_100/study/complete.json", {"step": 100})
    return output


def test_jsonl_tail_preserves_partial_utf8_and_does_not_replay(tmp_path):
    path = tmp_path / "metrics.jsonl"
    tail = JsonlTail(path)
    assert tail.read() == []
    record = json.dumps({"step": 1, "label": "λ"}, ensure_ascii=False).encode() + b"\n"
    split = record.index("λ".encode()) + 1
    path.write_bytes(record[:split])
    assert tail.read() == []
    assert tail.offset == 0
    with path.open("ab") as stream:
        stream.write(record[split:] + b"bad json\n" + b'{"step":2}\n')
    assert tail.read() == [{"step": 1, "label": "λ"}, {"step": 2}]
    assert tail.read() == []
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="Append-only"):
        tail.read()


def test_metrics_preserve_clocks_and_only_export_scalars(recorded):
    rows = []
    sink = SimpleNamespace(log=lambda values, step: rows.append((step, values)))
    metrics = Metrics(recorded, sink)
    assert metrics.poll()
    assert not metrics.poll()
    assert len(rows) == 4
    assert rows[0] == (33, {"trainer/loss/mean": -0.1, "trainer/source_time_unix": 123})
    assert rows[1][0] == 33
    assert rows[1][1]["staleness/age_min"] == rows[1][1]["staleness/age_max"] == 32
    assert rows[1][1]["rollout/consumed_reward_mean"] == 0.5
    assert rows[1][1]["queue/cohorts"] == 32
    assert rows[2][0] == 0
    assert rows[2][1]["generation/deferred/consumption_step"] == 33
    assert rows[3] == (100, {"checkpoint/complete": 1, "checkpoint/completed_step": 100})
    assert "private" not in json.dumps(rows)
    (recorded / "checkpoints/step_200/study").mkdir(parents=True)
    assert not metrics.poll()
    write_json(recorded / "checkpoints/step_200/study/complete.json", {"step": 200})
    assert metrics.poll()
    assert rows[-1][0] == 200


@pytest.mark.parametrize("status", ["finished", "crashed", "killed"])
def test_file_mode_import_preserves_status_and_scientific_config(recorded, status):
    write_json(recorded / "study-complete.json", {"step": 1000})
    write_json(recorded / "run-status.json", {"status": status})
    info = observe(recorded, once=True)
    storage = Storage(recorded / "tracking/runboard-runs")
    meta = storage.read_meta(info["project"], info["run_id"])
    rows, _ = storage.read_rows(info["project"], info["run_id"])
    assert len(rows) == 4
    assert meta["status"] == status
    assert meta["config"]["lag"] == 32
    assert meta["config"]["weight_decay"] == 0
    assert meta["config"]["checkpoint_interval"] == 100
    assert meta["config"]["intermediate_evaluation"] is False
    assert "dataset_path" not in meta["config"]


def test_http_backend_receives_exact_run_and_all_rows(recorded, tmp_path, monkeypatch):
    server = Server(tmp_path / "backend", "integration-token", host="127.0.0.1", port=0).start()
    try:
        monkeypatch.setenv("RUNBOARD_SERVER", f"http://127.0.0.1:{server.port}")
        monkeypatch.setenv("RUNBOARD_TOKEN", "integration-token")
        write_json(recorded / "run-status.json", {"status": "finished"})
        info = observe(recorded, once=True)
        query = f"/api/metrics?project={info['project']}&run={info['run_id']}"
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.port}{query}",
            headers={"Authorization": "Bearer integration-token"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.load(response)
        assert info["mode"] == "http"
        assert len(result["rows"]) == 4
        meta = Storage(tmp_path / "backend").read_meta(info["project"], info["run_id"])
        assert meta["status"] == "finished"
        assert "integration-token" not in json.dumps(meta)
    finally:
        server.stop()


def test_network_failure_spools_rows_without_raising(recorded, monkeypatch):
    import runboard.client as client

    def offline(*args, **kwargs):
        raise OSError("offline test")

    finish = client.Run.finish
    monkeypatch.setattr(client, "post_json", offline)
    monkeypatch.setattr(client.urllib.request, "urlopen", offline)
    monkeypatch.setattr(
        client.Run, "finish", lambda self, status="finished", timeout=2: finish(self, status, timeout=0)
    )
    monkeypatch.setenv("RUNBOARD_SERVER", "http://127.0.0.1:9")
    monkeypatch.setenv("RUNBOARD_TOKEN", "integration-token")
    write_json(recorded / "run-status.json", {"status": "finished"})
    info = observe(recorded, once=True)
    spools = [json.loads(p.read_text()) for p in client.spool_dir().glob("*.jsonl")]
    assert sum(len(item["rows"]) for item in spools) == 4
    assert all(item["run_id"] == info["run_id"] for item in spools)


def test_once_does_not_mark_an_active_run_finished(recorded):
    with pytest.raises(ValueError, match="--once"):
        observe(recorded, once=True)


def test_sidecar_waits_for_launcher_and_drains_final_trainer_metrics(recorded):
    write_json(recorded / "study-complete.json", {"step": 1000})
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "deepseek_study.tracking.runboard",
            str(recorded),
            "--parent-pid",
            str(os.getpid()),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not list((recorded / "tracking").glob("runboard-*.json")) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert process.poll() is None
        append(recorded / "metrics.jsonl", {"producer": "trainer", "step": 1000, "optim/lr": 1e-6})
        write_json(recorded / "run-status.json", {"status": "finished"})
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stderr
        info = json.loads(stdout)
        storage = Storage(recorded / "tracking/runboard-runs")
        rows, _ = storage.read_rows(info["project"], info["run_id"])
        assert len(rows) == 5
        assert any(row.get("trainer/optim/lr") == 1e-6 and row["_step"] == 1000 for row in rows)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_launcher_observer_failure_does_not_fail_training(study, tmp_path, monkeypatch, capsys):
    import torch
    from deepseek_study.runtime import launcher

    monkeypatch.setattr(launcher, "verify_upstream", lambda root: None)
    monkeypatch.setattr(launcher, "validate_prepared", lambda study: {})
    monkeypatch.setattr(launcher, "capture", lambda root, study: {"sha256": "source", "runtime": {"vllm": "test"}})
    monkeypatch.setattr(launcher, "snapshot", lambda root, dest, identity: (dest / "src").mkdir(parents=True))
    monkeypatch.setattr(launcher, "build", lambda *args: None)
    monkeypatch.setattr(launcher, "env_servers", lambda config: ())
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "mock")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(total_memory=1))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 0))
    monkeypatch.setattr(launcher.os, "killpg", lambda *args: None)

    class Process:
        pid = 12345

        def __init__(self, args, **kwargs):
            self.code = None
            if "deepseek_study.tracking.runboard" in args:
                self.code = 1
            if "controller" in args:
                write_json(study.output_dir / "study-complete.json", {"step": study.max_steps})
                self.code = 0
            if "deepseek_study.runtime.trainer" in args:
                self.code = 0

        def poll(self):
            return self.code

        def wait(self, timeout=None):
            return self.code or 0

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    assert launcher.launch(study, tmp_path) == study.output_dir
    assert json.loads((study.output_dir / "run-status.json").read_text())["status"] == "finished"
    assert "training continues" in capsys.readouterr().err


def test_live_dashboard_waits_for_final_paper_metrics(recorded):
    write_json(recorded / "paper-observer.json", {"enabled": True})
    process = subprocess.Popen(
        [sys.executable, "-m", "deepseek_study.tracking.runboard", str(recorded), "--parent-pid", str(os.getpid())],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not list((recorded / "tracking").glob("runboard-*.json")) and time.monotonic() < deadline:
            time.sleep(0.05)
        write_json(recorded / "run-status.json", {"status": "finished"})
        time.sleep(1.2)
        assert process.poll() is None
        append(recorded / "paper-metrics.jsonl", {"step": 1000, "metrics": {"mismatch/m2": 0.031}})
        write_json(recorded / "paper-status.json", {"status": "complete"})
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stderr
        info = json.loads(stdout)
        rows, _ = Storage(recorded / "tracking/runboard-runs").read_rows(info["project"], info["run_id"])
        assert any(row.get("paper/mismatch/m2") == 0.031 and row["_step"] == 1000 for row in rows)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
