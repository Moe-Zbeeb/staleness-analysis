import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from deepseek_study.rollouts.queue import QueueState
from deepseek_study.runtime import checkpoints


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("local_resume_control", SCRIPTS / "node_local_run.py")
LOCAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOCAL)


@pytest.fixture
def migration(tmp_path, study):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    study = study.model_copy(update={"lag": 0, "output_dir": workspace / "run"})
    LOCAL.atomic_json(workspace / "study.json", study.model_dump(mode="json"))
    source = tmp_path / "shared/step_5"
    for name in ("trainer/.metadata", "trainer/rank_0.distcp", "orchestrator/progress.pt", "rng/rank_0.pt"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original state")
    identity_hash = "a" * 64
    components_hash = checkpoints.seal(source, 1)
    state = QueueState(lag=0, completed_steps=5, generated_cohorts=5)
    checkpoints.save(source / "study", state, study.fingerprint(), identity_hash, components_hash)
    marker = json.loads((source / "study/complete.json").read_text())
    receipt = tmp_path / "migration.json"
    audit = {
        "format": 1,
        "status": "verified",
        "target_checkpoint": str(workspace / "resume/step_5"),
        "checkpoint_step": 5,
        "config_sha256": study.fingerprint(),
        "identity_sha256": identity_hash,
        "complete_sha256": LOCAL.digest(source / "study/complete.json"),
        "components_sha256": marker["components_sha256"],
        "queue_sha256": marker["queue_sha256"],
        "files": {
            str(path.relative_to(source)): {"size": path.stat().st_size, "sha256": LOCAL.digest(path)}
            for path in source.rglob("*") if path.is_file()
        },
    }
    LOCAL.atomic_json(receipt, audit)
    spec = {
        "workspace": str(workspace),
        "resume_checkpoint": str(source),
        "resume_receipt": str(receipt),
        "resume_expected_receipt_sha256": LOCAL.digest(receipt),
    }
    return spec, workspace, source, audit, study


def test_fresh_launch_remains_fresh_but_cannot_discard_sealed_resume(tmp_path):
    assert LOCAL.stage_resume({}, tmp_path) is None
    assert LOCAL.validate_resume_files({}, tmp_path, {}) is None
    with pytest.raises(ValueError, match="silently discarded"):
        LOCAL.validate_resume_files({}, tmp_path, {"resume": {}})


@pytest.mark.parametrize("field", ["resume_checkpoint", "resume_receipt", "resume_expected_receipt_sha256"])
def test_partial_resume_contract_fails_before_staging(migration, field):
    spec, workspace, *_ = migration
    del spec[field]
    with pytest.raises(ValueError, match="audited receipt"):
        LOCAL.stage_resume(spec, workspace)
    assert not (workspace / "resume").exists()


def test_resume_copy_preserves_original_and_cannot_overwrite(migration):
    spec, workspace, source, audit, _ = migration
    sealed = LOCAL.stage_resume(spec, workspace)
    target = LOCAL.validate_resume_files(spec, workspace, {"resume": sealed})
    assert target != source
    LOCAL.verify_resume_tree(source, audit["files"])
    LOCAL.verify_resume_tree(target, audit["files"])
    with pytest.raises(FileExistsError, match="replace"):
        LOCAL.stage_resume(spec, workspace)


@pytest.mark.parametrize("tamper", ["same_size_shard", "missing_file", "extra_file", "symlink"])
def test_resume_rejects_checkpoint_tampering_including_trainer_shards(migration, tamper):
    spec, workspace, source, *_ = migration
    if tamper == "same_size_shard":
        (source / "trainer/rank_0.distcp").write_bytes(b"modified state")
    elif tamper == "missing_file":
        (source / "rng/rank_0.pt").unlink()
    elif tamper == "extra_file":
        (source / "unexpected").write_bytes(b"new")
    else:
        (source / "alias").symlink_to(source / "trainer/.metadata")
    with pytest.raises(ValueError, match="checksum|inventory|symlink"):
        LOCAL.stage_resume(spec, workspace)
    assert not (workspace / "resume").exists()


def test_resume_rejects_changed_audit_and_outside_destination(migration):
    spec, workspace, _, audit, _ = migration
    receipt = Path(spec["resume_receipt"])
    audit["target_checkpoint"] = str(workspace.parent / "elsewhere")
    LOCAL.atomic_json(receipt, audit)
    with pytest.raises(ValueError, match="checksum"):
        LOCAL.stage_resume(spec, workspace)
    spec["resume_expected_receipt_sha256"] = LOCAL.digest(receipt)
    with pytest.raises(ValueError, match="another local"):
        LOCAL.stage_resume(spec, workspace)


def runtime_fixture(migration, monkeypatch):
    spec, workspace, _, _, study = migration
    sealed = LOCAL.stage_resume(spec, workspace)
    LOCAL.atomic_json(workspace / "ready.json", {"resume": sealed, "release": str(workspace / "release")})
    module = ModuleType("deepseek_study.runtime.identity")
    module.capture = lambda root, configuration: {"sha256": "a" * 64}
    monkeypatch.setitem(sys.modules, "deepseek_study.runtime.identity", module)
    return spec, workspace, study, module


def test_runtime_validation_uses_real_queue_and_checkpoint_protocol(migration, monkeypatch):
    spec, workspace, _, _ = runtime_fixture(migration, monkeypatch)
    assert LOCAL.validate_resume_runtime(spec) == workspace / "resume/step_5"


@pytest.mark.parametrize("change", ["config", "identity", "shard", "receipt"])
def test_runtime_validation_rejects_changes_after_sealed_staging(migration, monkeypatch, change):
    spec, workspace, study, module = runtime_fixture(migration, monkeypatch)
    if change == "config":
        changed = study.model_copy(update={"learning_rate": 2e-6})
        LOCAL.atomic_json(workspace / "study.json", changed.model_dump(mode="json"))
    elif change == "identity":
        module.capture = lambda root, configuration: {"sha256": "b" * 64}
    elif change == "shard":
        (workspace / "resume/step_5/trainer/rank_0.distcp").write_bytes(b"modified state")
    else:
        (workspace / "resume/migration-receipt.json").write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        LOCAL.validate_resume_runtime(spec)


def prepared_fixture(migration, monkeypatch):
    spec, workspace, _, _, study = migration
    previous = {"workspace": str(workspace)}
    LOCAL.atomic_json(workspace / "storage-spec.json", previous)
    release = workspace / "release"
    LOCAL.atomic_json(release / "manifests/model.json", {"files": []})
    package = {"manifests/model.json": LOCAL.digest(release / "manifests/model.json")}
    LOCAL.atomic_json(release / "PACKAGE_SHA256.json", package)
    for path in (study.dataset_path, study.data_manifest, study.prepared_model_path / "tokenizer_config.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen asset")
    ready = {
        "node": "test-node",
        "study_sha256": LOCAL.digest(workspace / "study.json"),
        "control_sha256": {"storage-spec.json": LOCAL.digest(workspace / "storage-spec.json")},
        "release": str(release),
        "package_sha256": LOCAL.digest(release / "PACKAGE_SHA256.json"),
        "dataset_sha256": LOCAL.digest(study.dataset_path),
        "data_manifest_sha256": LOCAL.digest(study.data_manifest),
        "prepared_tokenizer_sha256": LOCAL.digest(study.prepared_model_path / "tokenizer_config.json"),
    }
    LOCAL.atomic_json(workspace / "ready.json", ready)
    monkeypatch.setenv("SLURMD_NODENAME", "test-node")
    control = workspace.parent / "control"
    control.mkdir()
    return spec, workspace, study, control


def test_attach_resume_to_prestaged_workspace_without_restaging_assets(migration, monkeypatch):
    spec, workspace, _, control = prepared_fixture(migration, monkeypatch)
    result = LOCAL.prepare_resume(spec, control)
    assert LOCAL.validate_resume_files(spec, workspace, result) == workspace / "resume/step_5"
    assert json.loads((workspace / "storage-spec.json").read_text()) == spec
    assert result["control_sha256"]["storage-spec.json"] == LOCAL.digest(workspace / "storage-spec.json")
    assert json.loads((control / "local-ready.json").read_text()) == result


@pytest.mark.parametrize("change", ["spec", "source", "dataset", "running", "node"])
def test_attach_resume_refuses_changed_staging_or_running_workspace(migration, monkeypatch, change):
    spec, workspace, study, control = prepared_fixture(migration, monkeypatch)
    if change == "spec":
        spec["run_name"] = "changed"
    elif change == "source":
        (workspace / "release/manifests/model.json").write_text("changed")
    elif change == "dataset":
        study.dataset_path.write_text("changed")
    elif change == "running":
        study.output_dir.mkdir()
    else:
        monkeypatch.setenv("SLURMD_NODENAME", "another-node")
    with pytest.raises((ValueError, FileExistsError)):
        LOCAL.prepare_resume(spec, control)
    assert not (workspace / "resume").exists()


def test_supervisor_passes_audited_local_checkpoint_to_frozen_cli(migration, monkeypatch):
    spec, workspace, _, control = prepared_fixture(migration, monkeypatch)
    ready = LOCAL.prepare_resume(spec, control)
    spec.update(runtime=str(workspace / "runtime"), backup="/shared/new-run", metrics="/shared/new-metrics")
    LOCAL.atomic_json(workspace / "storage-spec.json", spec)
    ready["control_sha256"]["storage-spec.json"] = LOCAL.digest(workspace / "storage-spec.json")
    ready["python"] = "/prepared/python"
    LOCAL.atomic_json(workspace / "ready.json", ready)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(LOCAL, "local_environment", lambda *args: {})
    monkeypatch.setattr(LOCAL, "select_local_devices", lambda *args: ["0", "1", "2", "3"])
    monkeypatch.setattr(LOCAL.signal, "signal", lambda *args: None)
    commands = []
    processes = []

    def command(arguments, **kwargs):
        arguments = [str(item) for item in arguments]
        commands.append(arguments)
        if "--output" in arguments:
            LOCAL.atomic_json(Path(arguments[arguments.index("--output") + 1]), {})
        if "--receipt" in arguments:
            LOCAL.atomic_json(
                Path(arguments[arguments.index("--receipt") + 1]),
                {"world_size": 4, "devices": [{}, {}, {}, {}]},
            )

    class Process:
        pid = 999999
        returncode = 0

        def __init__(self, arguments, **kwargs):
            processes.append(arguments)

        def poll(self):
            return 0

        def wait(self, **kwargs):
            return 0

    monkeypatch.setattr(LOCAL, "command", command)
    monkeypatch.setattr(LOCAL.subprocess, "Popen", Process)
    LOCAL.run(spec)
    assert commands[0] == [
        "/prepared/python", str(workspace / "node_local_run.py"), "resume-check", "--control", str(workspace)
    ]
    assert processes[1] == [
        "/prepared/python", "-m", "deepseek_study.cli", "run", str(workspace / "study.json"),
        "--resume", str(workspace / "resume/step_5"),
    ]
    supervisor = json.loads((workspace / "supervisor.json").read_text())
    assert supervisor["resume_from"] == str(workspace / "resume/step_5")
