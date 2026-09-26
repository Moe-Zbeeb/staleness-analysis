import json

import pytest

from deepseek_study.audit import audit


def write_run(path, lag, starting, end):
    (path / "configs").mkdir()
    (path / "configs/protocol.json").write_text(json.dumps({"lag": lag, "max_steps": end, "responses_per_update": 2}))
    (path / "run.json").write_text(json.dumps({"starting_step": starting, "identity_sha256": "source"}))
    rows = [
        {
            "step": step,
            "learner_version": step - 1,
            "behavior_version": step - 1 if step <= lag else step - 1 - lag,
            "age_min": 0 if step <= lag else lag,
            "age_max": 0 if step <= lag else lag,
            "warmup": step <= lag,
            "response_ids": [f"{step}-a", f"{step}-b"],
            "responses": 2,
        }
        for step in range(starting + 1, end + 1)
    ]
    (path / "updates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (path / "study-complete.json").write_text(json.dumps({"step": end, "lag": lag, "identity_sha256": "source"}))


@pytest.mark.parametrize("lag,start,end", [(0, 0, 3), (8, 5, 11), (32, 35, 39)])
def test_auditor_handles_requested_lag_and_resumed_logs(tmp_path, lag, start, end):
    write_run(tmp_path, lag, start, end)
    result = audit(tmp_path)
    assert result["requested_lag"] == lag
    assert result["exact_staleness_updates"] == sum(step > lag for step in range(start + 1, end + 1))
    assert result["complete"]


def test_auditor_rejects_a_premature_completion_marker(tmp_path):
    write_run(tmp_path, 1, 0, 3)
    path = tmp_path / "updates.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError, match="Completion marker"):
        audit(tmp_path)
