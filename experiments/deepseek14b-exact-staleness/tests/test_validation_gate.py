import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def gate(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("validate_then_run", scripts / "validate_then_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module.validate_case
    for name in ("benchmark_1p5b", "launch_full_run", "local_backup", "node_local_run", "run_bounded_profile"):
        sys.modules.pop(name, None)


def write_case(path, mutation=None):
    (path / "run").mkdir()
    (path / "timing-summary.json").write_text(json.dumps({"status": "completed", "completed_updates": 2}))
    updates = [
        {
            "step": step,
            "responses": 512,
            "warmup": True,
            "age_min": 0,
            "age_max": 0,
            "step_wall_seconds": 100.0,
            "mean_reward": 0.4,
            "zero_advantage_fraction": 0.5,
        }
        for step in (1, 2)
    ]
    metrics = [{"step": step, "producer": "trainer", "optim/grad_norm": 0.1} for step in (1, 2)]
    if mutation:
        mutation(updates, metrics)
    for name, data in (("updates", updates), ("metrics", metrics)):
        (path / "run" / (name + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in data))


def test_complete_signal_and_timing_required(gate, tmp_path):
    write_case(tmp_path)
    assert gate(tmp_path)["mean_update_seconds"] == 100


@pytest.mark.parametrize("failure", ["nonfinite", "no_signal", "missing_gradient", "wrong_age", "extra_update"])
def test_validation_rejects_invalid_run(gate, tmp_path, failure):
    def mutate(updates, metrics):
        if failure == "nonfinite":
            metrics[1]["optim/grad_norm"] = float("nan")
        elif failure == "no_signal":
            updates[0]["zero_advantage_fraction"] = 1
        elif failure == "missing_gradient":
            metrics.pop()
        elif failure == "wrong_age":
            updates[0]["age_max"] = 1
        else:
            updates.append(updates[-1] | {"step": 3})

    write_case(tmp_path, mutate)
    with pytest.raises(ValueError):
        gate(tmp_path)
