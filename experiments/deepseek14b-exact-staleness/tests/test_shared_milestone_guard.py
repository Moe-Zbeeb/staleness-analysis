import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("milestone_control", SCRIPTS / "node_local_run.py")
LOCAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOCAL)
BACKUP = sys.modules["local_backup"]


@pytest.fixture
def guarded(tmp_path):
    workspace = tmp_path / "workspace"
    output, shared = workspace / "run", tmp_path / "shared"
    owner = {"run_uuid": "run", "config_sha256": "a" * 64, "identity_sha256": "b" * 64, "starting_step": 450}
    values = {
        "output_dir": str(output), "lag": 256, "max_steps": 1000,
        "checkpoint_keep_last": 3, "checkpoint_interval": 25, "checkpoint_keep_interval": 1000,
    }
    spec = {"workspace": str(workspace), "backup": str(shared), "require_verified_shared_milestones": True}
    LOCAL.atomic_json(output / "run.json", owner)
    LOCAL.atomic_json(workspace / "storage-spec.json", spec)
    LOCAL.atomic_json(workspace / "study.json", values)
    LOCAL.atomic_json(shared / "backup-owner.json", {
        **{k: owner[k] for k in ("run_uuid", "config_sha256", "identity_sha256")}, "source": str(output.resolve())
    })
    return spec, values, owner


def checkpoint(guarded, step, *, backup=False):
    spec, values, owner = guarded
    directory = Path(values["output_dir"]) / "checkpoints" / f"step_{step}"
    for name in ("trainer/.metadata", "trainer/shard.distcp", "orchestrator/progress.pt", "rng/rank_0.pt", "study/queue.pkl"):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original")
    components = [
        {"path": str(path.relative_to(directory)), "size": path.stat().st_size, "sha256": LOCAL.digest(path)}
        for path in directory.rglob("*") if path.is_file() and path.parent.name != "study"
    ]
    LOCAL.atomic_json(directory / "study/components.json", components)
    marker = {
        "format": 2, "step": step, "lag": 256,
        "config_sha256": owner["config_sha256"], "identity_sha256": owner["identity_sha256"],
        "components_sha256": LOCAL.digest(directory / "study/components.json"),
        "queue_sha256": LOCAL.digest(directory / "study/queue.pkl"),
    }
    LOCAL.atomic_json(directory / "study/complete.json", marker)
    if backup:
        BACKUP.backup_checkpoint(directory, Path(spec["backup"]) / "checkpoints" / f"step_{step}", owner)
    return directory


def test_guard_is_disabled_by_default(guarded):
    spec, values, _ = guarded
    spec.pop("require_verified_shared_milestones")
    values["checkpoint_keep_last"] = 4
    checkpoint(guarded, 550)
    assert LOCAL.SharedMilestoneGuard(spec, values).check() == []


@pytest.mark.parametrize("field,value", [("checkpoint_keep_last", 4), ("checkpoint_interval", 100),
                                         ("checkpoint_keep_interval", 100), ("max_steps", 999)])
def test_guard_rejects_a_different_retention_schedule(guarded, field, value):
    spec, values, _ = guarded
    values[field] = value
    with pytest.raises(ValueError, match="three local checkpoints"):
        LOCAL.SharedMilestoneGuard(spec, values)


def test_guard_waits_until_fifty_updates_then_fails_closed(guarded):
    spec, values, _ = guarded
    checkpoint(guarded, 500)
    checkpoint(guarded, 525)
    guard = LOCAL.SharedMilestoneGuard(spec, values)
    assert guard.check() == []
    checkpoint(guarded, 550)
    with pytest.raises(ValueError, match="no verified shared backup"):
        guard.check()
    assert not guard.verified


def test_guard_ignores_partial_checkpoints_and_prior_run_milestones(guarded):
    spec, values, _ = guarded
    checkpoint(guarded, 500)
    checkpoint(guarded, 525)
    (Path(values["output_dir"]) / "checkpoints/step_550/trainer").mkdir(parents=True)
    assert LOCAL.SharedMilestoneGuard(spec, values).check() == []


def test_guard_verifies_real_shared_files_once_and_caches_receipt(guarded, monkeypatch):
    spec, values, _ = guarded
    checkpoint(guarded, 500, backup=True)
    checkpoint(guarded, 525)
    checkpoint(guarded, 550)
    guard = LOCAL.SharedMilestoneGuard(spec, values)
    assert guard.check() == [500]
    receipt = Path(spec["backup"]) / "checkpoints/step_500/backup-verified.json"
    assert guard.verified[500] == LOCAL.digest(receipt)
    monkeypatch.setattr(guard, "verify_bounded", lambda marker: pytest.fail("A verified immutable backup was rehashed"))
    assert guard.check() == []


@pytest.mark.parametrize("change", ["same_size_file", "wrong_owner", "wrong_receipt", "missing_file", "extra_file", "symlink"])
def test_guard_rejects_invalid_shared_backups(guarded, change, tmp_path):
    spec, values, _ = guarded
    checkpoint(guarded, 500, backup=True)
    checkpoint(guarded, 550)
    shared = Path(spec["backup"])
    directory = shared / "checkpoints/step_500"
    if change == "same_size_file":
        (directory / "trainer/shard.distcp").write_bytes(b"modified")
    elif change == "wrong_owner":
        value = json.loads((shared / "backup-owner.json").read_text())
        value["run_uuid"] = "foreign"
        LOCAL.atomic_json(shared / "backup-owner.json", value)
    elif change == "wrong_receipt":
        value = json.loads((directory / "backup-verified.json").read_text())
        value["checkpoint"]["config_sha256"] = "foreign"
        LOCAL.atomic_json(directory / "backup-verified.json", value)
    elif change == "missing_file":
        (directory / "trainer/shard.distcp").unlink()
    elif change == "extra_file":
        (directory / "extra").write_text("unexpected")
    else:
        path = directory / "trainer/shard.distcp"
        destination = tmp_path / "outside"
        path.rename(destination)
        path.symlink_to(destination)
    guard = LOCAL.SharedMilestoneGuard(spec, values)
    with pytest.raises(ValueError, match="shared verification failed"):
        guard.check()
    assert not guard.verified


def test_guard_rejects_foreign_local_checkpoint(guarded):
    spec, values, _ = guarded
    path = checkpoint(guarded, 525) / "study/complete.json"
    marker = json.loads(path.read_text())
    marker["identity_sha256"] = "foreign"
    LOCAL.atomic_json(path, marker)
    with pytest.raises(ValueError, match="foreign"):
        LOCAL.SharedMilestoneGuard(spec, values).check()


def test_missing_local_milestone_cannot_evade_backup_guard(guarded):
    spec, values, _ = guarded
    checkpoint(guarded, 550)
    with pytest.raises(ValueError, match="disappeared locally"):
        LOCAL.SharedMilestoneGuard(spec, values).check()


@pytest.mark.parametrize("cancelled", [False, True])
def test_verification_deadline_or_interrupt_does_not_wait_for_an_unkillable_child(guarded, monkeypatch, cancelled):
    spec, values, _ = guarded
    guard = LOCAL.SharedMilestoneGuard(spec, values)
    guard.cancelled = lambda: cancelled
    clock = iter([0, 301])
    monkeypatch.setattr(LOCAL.time, "monotonic", lambda: next(clock))

    class Child:
        pid = 123456
        stdin = io.BytesIO()
        stdout = io.BytesIO()
        killed = False

        def poll(self):
            return None

        def kill(self):
            self.killed = True

        def wait(self, *args, **kwargs):
            pytest.fail("Timed-out verification must not block waiting for the child")

        def communicate(self, *args, **kwargs):
            pytest.fail("Timed-out verification must not communicate with the child")

    child = Child()
    monkeypatch.setattr(LOCAL.subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(RuntimeError, match="supervisor termination" if cancelled else "SIGKILL without waiting"):
        guard.verify_bounded({"step": 500})
    assert child.killed


@pytest.mark.parametrize("flag", [None, False])
def test_migration_requiring_backup_protection_cannot_disable_guard(flag):
    spec = {} if flag is None else {"require_verified_shared_milestones": flag}
    audit = {"local_retention_change": {"requires_shared_milestone_verification_before_local_pruning": True}}
    with pytest.raises(ValueError, match="requires verified shared"):
        LOCAL.validate_resume_retention_guard(spec, audit)
