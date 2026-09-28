import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("local_backup", SCRIPTS / "local_backup.py")
BACKUP = importlib.util.module_from_spec(SPEC)
sys.modules["local_backup"] = BACKUP
SPEC.loader.exec_module(BACKUP)
SPEC = importlib.util.spec_from_file_location("node_local_run", SCRIPTS / "node_local_run.py")
LOCAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOCAL)


def test_relocation_preserves_every_scientific_field(tmp_path):
    baseline = {
        "lag": 256,
        "max_steps": 1000,
        "checkpoint_interval": 100,
        "trainer_gpus": 4,
        "model_path": "/old/model",
        "output_dir": "/old/run",
        "learning_rate": 1e-6,
        "metrics_mirror_root": "/old/mirror",
        "responses_per_prompt": 8,
    }
    result = LOCAL.relocate_study(baseline, tmp_path, "new-run")
    assert result["metrics_mirror_root"] is None
    assert result["output_dir"] == str(tmp_path / "runs/new-run")
    assert {k: v for k, v in result.items() if k not in LOCAL.PATH_FIELDS} == {
        k: v for k, v in baseline.items() if k not in LOCAL.PATH_FIELDS
    }


def test_cache_and_editable_environment_is_local(tmp_path, monkeypatch):
    monkeypatch.setattr(LOCAL, "socket_directory", lambda workspace: tmp_path / "ipc")
    monkeypatch.setenv("RUNBOARD_TOKEN", "unused")
    monkeypatch.setenv("PYTHONHOME", "/old/python")
    environment = LOCAL.local_environment(tmp_path / "runtime", tmp_path / "work", tmp_path / "release")
    assert "RUNBOARD_TOKEN" not in environment and "PYTHONHOME" not in environment
    assert all(environment[key] == "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    assert environment["DEEPSEEK_STUDY_RUNBOARD"] == "0"
    for key in ("CUDA_CACHE_PATH", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TMPDIR"):
        assert Path(environment[key]).is_relative_to(tmp_path)
        assert Path(environment[key]).is_dir()


def test_failed_verification_does_not_replace_previous_backup(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"new")
    target.write_bytes(b"previous")
    monkeypatch.setattr(BACKUP, "digest", lambda _: "corrupt")
    with pytest.raises(ValueError, match="checksum"):
        BACKUP.verified_copy(source, target)
    assert target.read_bytes() == b"previous"
    assert not list(tmp_path.glob("*.tmp"))


def fixture_checkpoint(path, owner):
    for name, body in (
        ("trainer/shard.bin", b"weights"),
        ("trainer/.metadata", b"metadata"),
        ("orchestrator/progress.pt", b"sampler"),
        ("rng/rank_0.pt", b"rng"),
        ("study/queue.pkl", b"pending cohorts"),
    ):
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    components = [
        {"path": str(p.relative_to(path)), "size": p.stat().st_size}
        for p in path.rglob("*")
        if p.is_file() and p.parent.name != "study"
    ]
    BACKUP.atomic_json(path / "study/components.json", components)
    BACKUP.atomic_json(
        path / "study/complete.json",
        {
            "format": 2,
            "step": 100,
            "lag": 256,
            "identity_sha256": owner["identity_sha256"],
            "config_sha256": owner["config_sha256"],
            "queue_sha256": BACKUP.digest(path / "study/queue.pkl"),
            "components_sha256": BACKUP.digest(path / "study/components.json"),
        },
    )


def test_checkpoint_publication_hashes_trainer_and_queue(tmp_path):
    source, target = tmp_path / "local/step_100", tmp_path / "shared/step_100"
    owner = {"run_uuid": "run", "identity_sha256": "identity", "config_sha256": "config"}
    fixture_checkpoint(source, owner)
    assert BACKUP.backup_checkpoint(source, target, owner)
    receipt = json.loads((target / "backup-verified.json").read_text())
    assert receipt["files"]["trainer/shard.bin"]["sha256"] == BACKUP.digest(source / "trainer/shard.bin")
    assert receipt["files"]["study/queue.pkl"]["sha256"] == BACKUP.digest(source / "study/queue.pkl")
    assert not BACKUP.backup_checkpoint(source, target, owner)


def test_failed_checkpoint_never_publishes_complete_directory(tmp_path, monkeypatch):
    source, target = tmp_path / "local/step_100", tmp_path / "shared/step_100"
    owner = {"run_uuid": "run", "identity_sha256": "identity", "config_sha256": "config"}
    fixture_checkpoint(source, owner)
    original = BACKUP.verified_copy

    def interrupted(source, destination):
        if source.name == "shard.bin":
            raise OSError("shared storage disconnected")
        return original(source, destination)

    monkeypatch.setattr(BACKUP, "verified_copy", interrupted)
    with pytest.raises(OSError):
        BACKUP.backup_checkpoint(source, target, owner)
    assert not target.exists()
    assert not list(target.parent.glob(".incoming-*"))
    assert (source / "study/complete.json").is_file()


def test_incomplete_checkpoint_and_symlink_are_rejected(tmp_path):
    source = tmp_path / "local"
    source.mkdir()
    assert not BACKUP.backup_checkpoint(source, tmp_path / "shared", {})
    real = tmp_path / "real"
    real.write_bytes(b"secret")
    link = source / "link"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="regular file"):
        BACKUP.verified_copy(link, tmp_path / "copied")


def test_incremental_backup_preserves_run_identity_and_metrics(tmp_path):
    source, destination, metrics = [tmp_path / x for x in ("local", "nfs", "xfs")]
    source.mkdir()
    owner = {"run_uuid": "run", "identity_sha256": "identity", "config_sha256": "config"}
    BACKUP.atomic_json(source / "run.json", owner)
    journal = source / "updates.jsonl"
    journal.write_text('{"step":1}\n')
    backup = BACKUP.Backup(source, destination, metrics)
    backup.poll()
    with journal.open("a") as stream:
        stream.write('{"step":2}\n')
    backup.poll()
    assert (destination / "updates.jsonl").read_bytes() == journal.read_bytes()
    assert (metrics / "updates.jsonl").read_bytes() == journal.read_bytes()
    BACKUP.atomic_json(source / "run.json", {**owner, "run_uuid": "different"})
    with pytest.raises(ValueError, match="another run"):
        backup.poll()


def test_overlapping_backups_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        BACKUP.Backup(tmp_path, tmp_path / "backup")


def test_shared_checkpoint_restores_exact_pending_cohorts(tmp_path):
    from deepseek_study.rollouts.queue import Cohort, QueueState
    from deepseek_study.runtime import checkpoints

    owner = {"run_uuid": "run", "identity_sha256": "identity", "config_sha256": "config"}
    source, target = tmp_path / "local/step_2", tmp_path / "shared/step_2"
    fixture_checkpoint(source, owner)
    cohorts = {
        v: Cohort(v, (f"response-{v}",), 10, {"original_behavior_logprobs": [-0.3, -0.8]}, f"digest-{v}")
        for v in range(2)
    }
    state = QueueState(
        lag=256,
        generation_stop=744,
        completed_steps=2,
        pending=cohorts,
        generated_ids={"response-0", "response-1"},
        generated_cohorts=4,
        generated_output_tokens=40,
    )
    components = checkpoints.seal(source, 1)
    checkpoints.save(source / "study", state, "config", "identity", components)
    BACKUP.backup_checkpoint(source, target, owner)
    checkpoints.verify_components(target)
    restored = checkpoints.load(target / "study", "config", "identity")
    assert restored == state
    restored.validate()


@pytest.mark.parametrize("external_dependency", [False, True])
def test_runtime_relocation_rewrites_entrypoints_and_fails_on_external_dependencies(tmp_path, external_dependency):
    runtime, shared, release = [tmp_path / name for name in ("runtime", "shared-prime", "release")]
    source_python = tmp_path / "shared-python/bin/python3.12"
    source_python.parent.mkdir(parents=True)
    source_python.write_text("binary-placeholder")
    (shared / ".venv/bin").mkdir(parents=True)
    (shared / ".venv/bin/python").symlink_to(source_python)
    venv = runtime / "prime-rl/.venv"
    site = venv / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    (venv / "bin").mkdir()
    (venv / "pyvenv.cfg").write_text(f"home = {source_python.parent}\ninclude-system-site-packages = false\n")
    (venv / "bin/ninja").write_text(f"#!{shared}/.venv/bin/python\nprint('entrypoint')\n")
    (site / "prime.pth").write_text(str(shared / "src") + "\n")
    (site / "_editable_impl_deepseek_staleness_study.pth").write_text("/mnt/nfs/old-release/src\n")
    if external_dependency:
        (site / "unknown.pth").write_text("/mnt/nfs/unmapped-package\n")
    BACKUP.atomic_json(runtime / "PRESTAGED_RUNTIME.json", {"source": str(shared)})
    if external_dependency:
        with pytest.raises(ValueError, match="still references shared"):
            LOCAL.relocate_runtime(runtime, shared, release)
    else:
        python = LOCAL.relocate_runtime(runtime, shared, release)
        assert python.is_symlink()
        assert str(runtime / "python/bin") in (venv / "pyvenv.cfg").read_text()
        assert str(venv / "bin/python") in (venv / "bin/ninja").read_text()
        assert (site / "prime.pth").read_text() == str(runtime / "prime-rl/src") + "\n"
        assert (site / "_editable_impl_deepseek_staleness_study.pth").read_text() == str(release / "src") + "\n"


@pytest.mark.parametrize("external_namespace", [False, True])
def test_runtime_audit_supports_namespace_packages_and_rejects_remote_paths(tmp_path, monkeypatch, external_namespace):
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "python"))
    monkeypatch.setattr(sys, "argv", ["audit", str(tmp_path)])
    for name in ("prime_rl", "torch", "vllm", "verifiers"):
        module = types.ModuleType(name)
        if name == "prime_rl":
            module.__file__ = None
            module.__path__ = ["/mnt/nfs/prime-rl/src/prime_rl" if external_namespace else str(tmp_path / name)]
        else:
            module.__file__ = str(tmp_path / name / "__init__.py")
        monkeypatch.setitem(sys.modules, name, module)
    if external_namespace:
        with pytest.raises(RuntimeError, match="outside local storage"):
            exec(LOCAL.RUNTIME_AUDIT, {})
    else:
        exec(LOCAL.RUNTIME_AUDIT, {})


def test_repository_copy_preserves_tracked_nested_monitor(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    monitor = source / "src/prime_rl/monitors/wandb/monitor.py"
    monitor.parent.mkdir(parents=True)
    monitor.write_text("official-monitor")
    (source / "wandb").mkdir()
    (source / "wandb/runtime-log").write_text("runtime")
    LOCAL.copy_tree(source, destination, LOCAL.REPOSITORY_EXCLUDES)
    assert (destination / "src/prime_rl/monitors/wandb/monitor.py").read_text() == "official-monitor"
    assert not (destination / "wandb").exists()


def test_local_git_rejects_wrong_archive_before_extracting(tmp_path):
    (tmp_path / "git-lfs-3.7.1.tar.gz").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        LOCAL.configure_local_git(tmp_path)
    assert not (tmp_path / "prime-rl/.venv/bin/git-lfs").exists()


def test_socket_paths_fit_linux_unix_domain_limit():
    path = LOCAL.socket_directory(Path("/tmp") / ("long-workspace-" * 15))
    address = path / "4af5fcfb-7ecd-4e1f-b72a-863e7c9a295d"
    assert len(str(address).encode()) <= 107
    assert path != LOCAL.socket_directory("/tmp/another-workspace")


def test_shared_retention_only_removes_verified_owned_checkpoints(tmp_path):
    owner = {"run_uuid": "run", "identity_sha256": "identity", "config_sha256": "config"}
    for step in (1, 5, 10, 100, 105, 110):
        source = tmp_path / "local" / f"step_{step}"
        target = tmp_path / "shared" / f"step_{step}"
        fixture_checkpoint(source, owner)
        marker = json.loads((source / "study/complete.json").read_text())
        marker["step"] = step
        BACKUP.atomic_json(source / "study/complete.json", marker)
        assert BACKUP.backup_checkpoint(source, target, owner)
    (tmp_path / "shared/step_115").mkdir()
    assert BACKUP.prune_verified_checkpoints(tmp_path / "shared", owner) == [1, 5, 10]
    assert sorted(p.name for p in (tmp_path / "shared").iterdir()) == ["step_100", "step_105", "step_110", "step_115"]
    with pytest.raises(ValueError, match="owners"):
        BACKUP.prune_verified_checkpoints(tmp_path / "shared", {**owner, "run_uuid": "other"})


def test_backup_does_not_recopy_pruned_local_checkpoints(tmp_path, monkeypatch):
    owner = {"run_uuid": "run", "identity_sha256": "identity", "config_sha256": "config"}
    source, shared = tmp_path / "local", tmp_path / "shared"
    source.mkdir()
    BACKUP.atomic_json(source / "run.json", owner)
    for step in (1, 5, 10, 15):
        path = source / "checkpoints" / f"step_{step}"
        fixture_checkpoint(path, owner)
        marker = json.loads((path / "study/complete.json").read_text())
        marker["step"] = step
        BACKUP.atomic_json(path / "study/complete.json", marker)
    backup = BACKUP.Backup(source, shared)
    backup.poll()
    assert sorted(p.name for p in (shared / "checkpoints").iterdir()) == ["step_10", "step_15"]
    original = BACKUP.backup_checkpoint

    def guarded(path, *args):
        assert path.name in {"step_10", "step_15"}
        return original(path, *args)

    monkeypatch.setattr(BACKUP, "backup_checkpoint", guarded)
    assert backup.poll()["new_checkpoints"] == []


@pytest.mark.parametrize("snapshot", [False, True])
def test_atomic_replacement_keeps_live_snapshot_but_rejects_checkpoint_change(tmp_path, monkeypatch, snapshot):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"original")
    original_fsync = BACKUP.os.fsync
    replaced = False

    def replace_after_copy(fd):
        nonlocal replaced
        original_fsync(fd)
        if not replaced:
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"new version")
            replacement.replace(source)
            replaced = True

    monkeypatch.setattr(BACKUP.os, "fsync", replace_after_copy)
    if snapshot:
        BACKUP.verified_copy(source, target, snapshot=True)
        assert target.read_bytes() == b"original"
        BACKUP.verified_copy(source, target, snapshot=True)
        assert target.read_bytes() == b"new version"
    else:
        with pytest.raises(ValueError, match="replaced"):
            BACKUP.verified_copy(source, target)
        assert not target.exists()


def test_live_snapshot_rejects_in_place_truncation(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"original")
    original_fsync = BACKUP.os.fsync

    def truncate_after_copy(fd):
        original_fsync(fd)
        source.write_bytes(b"short")

    monkeypatch.setattr(BACKUP.os, "fsync", truncate_after_copy)
    with pytest.raises(ValueError, match="truncated"):
        BACKUP.verified_copy(source, target, snapshot=True)
    assert not target.exists()


def test_active_backup_never_traverses_transient_policy_or_checkpoint_trees(tmp_path, monkeypatch):
    source, destination = tmp_path / "local", tmp_path / "nfs"
    source.mkdir()
    BACKUP.atomic_json(source / "run.json", {"run_uuid": "run", "identity_sha256": "identity", "config_sha256": "config"})
    for folder in ("historical-exports", "broadcasts", "checkpoints"):
        (source / folder / "disappearing").mkdir(parents=True)
    (source / "updates.jsonl").write_text('{"step":1}\n')
    original = BACKUP.os.scandir

    def guarded(path):
        if any(part in ("historical-exports", "broadcasts") for part in Path(path).parts):
            raise FileNotFoundError("Transient tree was traversed")
        return original(path)

    monkeypatch.setattr(BACKUP.os, "scandir", guarded)
    result = BACKUP.Backup(source, destination).poll()
    assert result["status"] == "verified"
    assert (destination / "updates.jsonl").is_file()
    assert not (destination / "historical-exports").exists()
