import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/continue_existing_run.py"
SPEC = importlib.util.spec_from_file_location("continuation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def checkpoint(root, index, **changes):
    directory = root / "checkpoints" / f"step_{index}" / "study"
    directory.mkdir(parents=True)
    value = {"format": 2, "step": index, "config_sha256": "config", "identity_sha256": "identity", **changes}
    (directory / "complete.json").write_text(json.dumps(value))
    return directory.parent


def test_continuation_uses_latest_complete_checkpoint_only(tmp_path):
    checkpoint(tmp_path, 100)
    selected = checkpoint(tmp_path, 200)
    (tmp_path / "checkpoints/step_300/study").mkdir(parents=True)
    assert MODULE.select_checkpoint(tmp_path, "config", "identity", 1000) == (200, selected)


@pytest.mark.parametrize("field", ["config_sha256", "identity_sha256", "step"])
def test_continuation_rejects_incompatible_checkpoint(tmp_path, field):
    checkpoint(tmp_path, 100, **{field: "changed"})
    with pytest.raises(ValueError, match="Incompatible"):
        MODULE.select_checkpoint(tmp_path, "config", "identity", 1000)


def test_continuation_never_restarts_from_scratch(tmp_path):
    with pytest.raises(ValueError, match="No complete checkpoint"):
        MODULE.select_checkpoint(tmp_path, "config", "identity", 1000)


def test_continuation_does_not_retrain_finished_budget(tmp_path):
    checkpoint(tmp_path, 1000)
    with pytest.raises(ValueError, match="already reached the budget"):
        MODULE.select_checkpoint(tmp_path, "config", "identity", 1000)
