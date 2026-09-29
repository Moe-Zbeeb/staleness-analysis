import importlib.util
import json
import pickle
import sys
from pathlib import Path

import pytest

from deepseek_study.rollouts.async_queue import AsyncQueueState
from deepseek_study.rollouts.queue import Cohort
from deepseek_study.runtime import checkpoints


SPEC = importlib.util.spec_from_file_location(
    "migrate_inference_checkpoint", Path(__file__).resolve().parents[1] / "scripts/migrate_inference_checkpoint.py"
)
MIGRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIGRATION)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def write_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def migration(tmp_path, study):
    assets = tmp_path / "assets"
    release = tmp_path / "release"
    records = []
    for name, data in (
        ("model.safetensors", b"weights"),
        ("tokenizer.json", b"vocabulary"),
        ("tokenizer_config.json", b"original metadata"),
    ):
        path = write_file(assets / "model" / name, data)
        records.append({"name": name, **MIGRATION.file_record(path)})
    model_manifest = write_json(
        release / "manifests/model.json", {"repo_id": "locked-model", "revision": "fixed", "files": records}
    )
    data = write_file(assets / "train.parquet", b"original questions")
    body = {"format": 1, "contract": {"source_sha256": MIGRATION.file_record(data)["sha256"]}, "records": []}
    manifest = write_json(assets / "train-manifest.json", {**body, "sha256": MIGRATION.object_digest(body)})
    native = assets / "native-model"
    write_file(native / "tokenizer_config.json", b"native metadata")
    for name in ("model.safetensors", "tokenizer.json"):
        (native / name).symlink_to(assets / "model" / name)
    receipt = {
        "model_id": "locked-model",
        "revision": "fixed",
        "dataset_sha256": body["contract"]["source_sha256"],
        "prepared_tokenizer_config_sha256": MIGRATION.file_record(native / "tokenizer_config.json")["sha256"],
        "files": records,
        "tokenizer_parity": {"probes_passed": 5},
    }
    write_json(native / "study-assets.json", receipt)
    write_file(release / "src/frozen.py", b"original source")
    lock = write_file(release / "vendor/prime-rl/uv.lock", b"original runtime lock")
    identity_body = {
        "format": 1,
        "source_files": {
            "manifests/model.json": MIGRATION.file_record(model_manifest)["sha256"],
            "src/frozen.py": MIGRATION.file_record(release / "src/frozen.py")["sha256"],
        },
        "lock_sha256": MIGRATION.file_record(lock)["sha256"],
        "runtime": {"torch": "frozen"},
        "data_sha256": MIGRATION.object_digest(body),
    }
    identity = {**identity_body, "sha256": MIGRATION.object_digest(identity_body)}
    source_config = study.model_copy(
        update={
            "trainer_gpus": 4,
            "inference_gpus": 5,
            "inference_tensor_parallel": 1,
            "lag": 256,
            "max_steps": 1000,
            "historical_rollouts": tmp_path / "source-history",
        }
    )
    target_config = source_config.model_copy(
        update={
            "inference_gpus": 4,
            "model_path": tmp_path / "future/assets/model",
            "dataset_path": tmp_path / "future/assets/train.parquet",
            "data_manifest": tmp_path / "future/assets/train-manifest.json",
            "prepared_model_path": tmp_path / "future/assets/native-model",
            "output_dir": tmp_path / "future/runs/target",
            "historical_rollouts": tmp_path / "target/history",
        }
    )
    baseline = target_config.model_copy(
        update={
            "model_path": assets / "model",
            "dataset_path": data,
            "prepared_model_path": native,
            "data_manifest": manifest,
        }
    )
    original = tmp_path / "shared/step_400"
    for name in (
        "trainer/.metadata",
        "orchestrator/progress.pt",
        *(f"trainer/__{rank}_0.distcp" for rank in range(4)),
        *(f"rng/rank_{rank}.pt" for rank in range(4)),
        "historical/sampler.pt",
    ):
        write_file(original / name, ("unchanged " + name).encode())
    pending = {
        version: Cohort(
            version,
            tuple(f"{version}:{index}" for index in range(source_config.response_batch_size)),
            100,
            {"payload": "unchanged"},
            "payload-digest",
        )
        for version in range(144, 400)
    }
    state = AsyncQueueState(
        256,
        generation_stop=744,
        completed_steps=400,
        pending=pending,
        generated_ids={identifier for cohort in pending.values() for identifier in cohort.response_ids},
        generated_cohorts=656,
    )
    checkpoints.save(
        original / "study", state, source_config.fingerprint(), identity["sha256"], checkpoints.seal(original, 4)
    )
    write_backup(original)
    arguments = {
        "source_checkpoint": original,
        "destination_checkpoint": tmp_path / "adapted/step_400",
        "source_study": write_json(tmp_path / "source-study.json", source_config.model_dump(mode="json")),
        "target_study": write_json(tmp_path / "target-study.json", target_config.model_dump(mode="json")),
        "target_baseline": write_json(tmp_path / "target-baseline.json", baseline.model_dump(mode="json")),
        "source_identity": write_json(tmp_path / "identity.json", identity),
        "target_release": release,
        "source_assets_receipt": write_json(tmp_path / "original-assets.json", receipt),
        "source_data_manifest": write_json(tmp_path / "original-manifest.json", json.loads(manifest.read_text())),
        "receipt_path": tmp_path / "audit/migration.json",
        "target_checkpoint": tmp_path / "future/resume/step_400",
    }
    return arguments


def write_backup(original):
    files = {
        name: {"bytes": value["size"], "sha256": value["sha256"]}
        for name, value in MIGRATION.inventory(original).items()
        if name != "backup-verified.json"
    }
    write_json(
        original / "backup-verified.json",
        {
            "run_uuid": "original-run",
            "checkpoint": json.loads((original / "study/complete.json").read_text()),
            "files": files,
        },
    )


def test_migration_preserves_optimizer_rng_queue_and_original_bytes(migration):
    before = MIGRATION.inventory(migration["source_checkpoint"])
    receipt = MIGRATION.migrate(**migration)
    assert MIGRATION.inventory(migration["source_checkpoint"]) == before
    after = MIGRATION.inventory(migration["destination_checkpoint"])
    assert {name for name in before if before[name] != after[name]} == {"study/complete.json"}
    assert receipt["files"] == after
    assert receipt["queue"]["ready_cohorts"] == 256
    assert receipt["queue"]["pending_jobs"] == 0
    assert receipt["target_checkpoint"] == str(migration["target_checkpoint"])
    assert receipt["source_complete"]["queue_sha256"] == receipt["queue_sha256"]
    checkpoints.verify_components(migration["destination_checkpoint"])
    restored = checkpoints.load(
        migration["destination_checkpoint"] / "study", receipt["config_sha256"], receipt["identity_sha256"]
    )
    assert restored.completed_steps == 400 and sorted(restored.pending) == list(range(144, 400))


@pytest.mark.parametrize(
    "field,value",
    [
        ("learning_rate", 2e-6),
        ("trainer_gpus", 3),
        ("response_max_tokens", 127),
        ("max_steps", 999),
        ("lag", 8),
        ("seed", 18),
        ("inference_tensor_parallel", 2),
    ],
)
def test_migration_rejects_scientific_and_trainer_changes(migration, field, value):
    target = json.loads(migration["target_study"].read_text())
    target[field] = value
    write_json(migration["target_study"], target)
    with pytest.raises(ValueError):
        MIGRATION.migrate(**migration)
    assert not migration["destination_checkpoint"].exists()
    assert not migration["receipt_path"].exists()


@pytest.mark.parametrize("filename", ["trainer/__0_0.distcp", "rng/rank_2.pt", "study/queue.pkl"])
def test_all_file_hashes_reject_same_size_corruption(migration, filename):
    path = migration["source_checkpoint"] / filename
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(ValueError, match="checksum"):
        MIGRATION.migrate(**migration)


def test_migration_rejects_pending_generation_jobs(migration):
    source = migration["source_checkpoint"]
    with (source / "study/queue.pkl").open("rb") as stream:
        state = pickle.load(stream)
    state.pending.pop(399)
    state.generated_cohorts -= 1
    state.jobs[399] = {"job": "not ready"}
    marker = json.loads((source / "study/complete.json").read_text())
    checkpoints.save(
        source / "study", state, marker["config_sha256"], marker["identity_sha256"], marker["components_sha256"]
    )
    write_backup(source)
    with pytest.raises(ValueError, match="pending historical"):
        MIGRATION.migrate(**migration)


def test_migration_accepts_correctly_shortened_queue_in_drain(migration):
    original = migration["source_checkpoint"]
    with (original / "study/queue.pkl").open("rb") as stream:
        old = pickle.load(stream)
    pending = {v: Cohort(v, tuple(f"{v}:{i}" for i in range(8)), 100, {}, "digest") for v in range(544, 744)}
    state = AsyncQueueState(
        256,
        generation_stop=744,
        completed_steps=800,
        pending=pending,
        generated_ids={i for c in pending.values() for i in c.response_ids},
        generated_cohorts=1000,
    )
    marker = json.loads((original / "study/complete.json").read_text())
    checkpoints.save(
        original / "study", state, marker["config_sha256"], marker["identity_sha256"], marker["components_sha256"]
    )
    source = original.with_name("step_800")
    original.rename(source)
    write_backup(source)
    migration.update(
        source_checkpoint=source,
        destination_checkpoint=migration["destination_checkpoint"].with_name("step_800"),
        target_checkpoint=migration["target_checkpoint"].with_name("step_800"),
    )
    receipt = MIGRATION.migrate(**migration)
    assert old.completed_steps == 400
    assert receipt["queue"]["ready_cohorts"] == 200


@pytest.mark.parametrize(
    "asset", ["model/model.safetensors", "native-model/tokenizer_config.json", "train.parquet", "train-manifest.json"]
)
def test_migration_rejects_changed_assets(migration, asset):
    baseline = json.loads(migration["target_baseline"].read_text())
    path = Path(baseline["model_path"]).parent / asset
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        MIGRATION.migrate(**migration)


def test_migration_rejects_checkpoint_symlink(migration, tmp_path):
    source = migration["source_checkpoint"]
    target = source / "trainer/__0_0.distcp"
    real = tmp_path / "outside.distcp"
    target.rename(real)
    target.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        MIGRATION.migrate(**migration)


def test_migration_refuses_overwrite_and_changed_frozen_source(migration):
    (migration["target_release"] / "src/frozen.py").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="Frozen training source"):
        MIGRATION.migrate(**migration)
    migration["destination_checkpoint"].mkdir(parents=True)
    with pytest.raises(FileExistsError):
        MIGRATION.migrate(**migration)


def test_migration_receipt_stages_and_validates_with_node_local_runner(migration, monkeypatch):
    from deepseek_study.runtime import identity

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("local_backup", "node_local_run"):
        module_spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
        module = importlib.util.module_from_spec(module_spec)
        monkeypatch.setitem(sys.modules, name, module)
        module_spec.loader.exec_module(module)
    local = sys.modules["node_local_run"]
    audit = MIGRATION.migrate(**migration)
    workspace = migration["target_checkpoint"].parents[1]
    workspace.mkdir()
    spec = {
        "workspace": str(workspace),
        "resume_checkpoint": str(migration["destination_checkpoint"]),
        "resume_receipt": str(migration["receipt_path"]),
        "resume_expected_receipt_sha256": MIGRATION.file_record(migration["receipt_path"])["sha256"],
    }
    resumed = local.stage_resume(spec, workspace)
    ready = {"release": str(migration["target_release"]), "resume": resumed}
    write_json(workspace / "ready.json", ready)
    write_json(workspace / "study.json", json.loads(migration["target_study"].read_text()))
    assert local.validate_resume_files(spec, workspace, ready) == migration["target_checkpoint"]
    monkeypatch.setattr(identity, "capture", lambda *args: audit["source_identity"])
    assert local.validate_resume_runtime(spec) == migration["target_checkpoint"]
    monkeypatch.setattr(identity, "capture", lambda *args: {"sha256": "changed-runtime"})
    with pytest.raises(ValueError, match="frozen runtime identity differs"):
        local.validate_resume_runtime(spec)
