import random
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deepseek_study.runtime import checkpoints, identity
from deepseek_study.rollouts.queue import QueueState
from deepseek_study.runtime.trainer_state import CheckpointWithRNG, capture_rng, restore_rng


def test_source_snapshot_detects_code_edits_and_freezes_original_bytes(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "vendor/prime-rl").mkdir(parents=True)
    (root / "vendor/prime-rl/uv.lock").write_text("locked")
    (root / "pyproject.toml").write_text("recipe")
    code = root / "src/run.py"
    code.write_text("x = 1\n")
    monkeypatch.setattr(identity, "load_manifest", lambda path: {"sha256": "prepared-data"})
    study = SimpleNamespace(data_manifest=tmp_path / "manifest")
    original = identity.capture(root, study)
    frozen = tmp_path / "frozen"
    identity.snapshot(root, frozen, original)
    assert identity.read_identity(frozen / "identity.json") == original
    code.write_text("x = 2\n")
    assert identity.capture(root, study)["sha256"] != original["sha256"]
    assert (frozen / "src/run.py").read_text() == "x = 1\n"
    with pytest.raises(ValueError, match="changed"):
        identity.snapshot(root, tmp_path / "other", original)


def test_checkpoint_completion_requires_all_components_and_rejects_truncation(tmp_path):
    for name in ("trainer/.metadata", "trainer/rank0.distcp", "orchestrator/progress.pt", "rng/rank_0.pt"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"complete")
    component_hash = checkpoints.seal(tmp_path, 1)
    checkpoints.save(tmp_path / "study", QueueState(0), "config", "source", component_hash)
    checkpoints.verify_components(tmp_path)
    (tmp_path / "trainer/rank0.distcp").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="incomplete"):
        checkpoints.verify_components(tmp_path)


def test_checkpoint_rejects_changed_sampler_content_even_at_same_size(tmp_path):
    for name in ("trainer/.metadata", "orchestrator/progress.pt", "rng/rank_0.pt"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original")
    component_hash = checkpoints.seal(tmp_path, 1)
    checkpoints.save(tmp_path / "study", QueueState(0), "config", "source", component_hash)
    (tmp_path / "orchestrator/progress.pt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        checkpoints.verify_components(tmp_path)


def test_trainer_random_states_resume_the_same_next_values():
    state = capture_rng()
    expected = random.random(), np.random.random(), torch.rand(3)
    restore_rng(state)
    actual = random.random(), np.random.random(), torch.rand(3)
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])


def test_retention_only_prunes_completed_checkpoints_after_a_new_commit(tmp_path):
    for step in (25, 50, 75, 100, 125, 150):
        directory = tmp_path / f"step_{step}" / "study"
        directory.mkdir(parents=True)
        (directory / "complete.json").write_text(json.dumps({"identity_sha256": "source", "step": step}))
    (tmp_path / "step_175").mkdir()
    assert checkpoints.prune_complete(tmp_path, 2, 100, "source") == [25, 50, 75]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["step_100", "step_125", "step_150", "step_175"]


def test_official_checkpoint_adapter_saves_and_restores_rng_with_rank_identity(tmp_path, monkeypatch):
    class Manager:
        world = SimpleNamespace(rank=0)

        def save(self, step):
            self.get_ckpt_path(step).mkdir(parents=True)

        def load(self, step, path=None):
            assert (path or self.get_ckpt_path(step)).is_dir()

        def get_ckpt_path(self, step):
            return tmp_path / f"step_{step}" / "trainer"

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    manager = CheckpointWithRNG(Manager())
    manager.save(3)
    expected = torch.rand(5)
    manager.load(3)
    assert torch.equal(torch.rand(5), expected)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="original sharding"):
        manager.load(3)
