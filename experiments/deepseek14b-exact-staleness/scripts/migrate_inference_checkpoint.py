import argparse
import hashlib
import json
import os
import pickle
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from deepseek_study.config import StudyConfig
from deepseek_study.rollouts.async_queue import AsyncQueueState


PATH_FIELDS = {
    "model_path",
    "dataset_path",
    "data_manifest",
    "prepared_model_path",
    "output_dir",
    "metrics_mirror_root",
    "historical_rollouts",
}


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_record(path):
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Not a regular file: {path}")
        result = hashlib.sha256()
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed while hashing: {path}")
    return {"size": before.st_size, "sha256": result.hexdigest()}


def read_json(path):
    file_record(path)
    return json.loads(Path(path).read_text())


def inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Checkpoint must be a regular directory")
    files = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError(f"Checkpoint contains a symlink or nonregular file: {path}")
        if stat.S_ISREG(mode):
            files[str(path.relative_to(root))] = file_record(path)
    return files


def safe_relative(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name or name in {"", "."}:
        raise ValueError(f"Unsafe relative path: {name}")
    return path


def validate_config_change(source, target, *, allow_local_retention_change=False):
    before = source.model_dump(mode="json")
    after = target.model_dump(mode="json")
    changes = {key: {"before": before[key], "after": after[key]} for key in before if before[key] != after[key]}
    if source.trainer_gpus != 4 or target.trainer_gpus != 4:
        raise ValueError("Migration must preserve all four trainer ranks")
    if source.inference_gpus != 5 or target.inference_gpus != 4:
        raise ValueError("Only the authorized inference GPU change from five to four is supported")
    if source.lag != 256 or source.historical_rollouts is None or target.historical_rollouts is None:
        raise ValueError("Migration requires the historical exact-lag256 study")
    if type(allow_local_retention_change) is not bool:
        raise ValueError("Local milestone retention override must be an explicit boolean")
    allowed = PATH_FIELDS | {"inference_gpus"}
    if allow_local_retention_change:
        if (
            source.checkpoint_keep_interval != 100
            or target.checkpoint_keep_interval != 1000
            or source.max_steps != 1000
            or target.max_steps != 1000
            or source.checkpoint_interval != 25
            or target.checkpoint_interval != 25
            or source.checkpoint_keep_last != 4
            or target.checkpoint_keep_last != 3
        ):
            raise ValueError(
                "Local milestone retention override only permits interval 100 to 1000 for a 1000-update run "
                "with checkpoints every 25 updates and local keep-last reduced from four to three"
            )
        allowed.update({"checkpoint_keep_interval", "checkpoint_keep_last"})
    forbidden = set(changes) - allowed
    if forbidden:
        raise ValueError(f"Migration changes forbidden scientific fields: {sorted(forbidden)}")
    for config in (before, after):
        for key in PATH_FIELDS:
            if config[key] is not None and not Path(config[key]).is_absolute():
                raise ValueError(f"Migration paths must be absolute: {key}")
    return changes


def validate_source_identity(path, target_release):
    identity = read_json(path)
    if identity.get("format") != 1 or identity.get("sha256") != object_digest(
        {key: value for key, value in identity.items() if key != "sha256"}
    ):
        raise ValueError("Original source/runtime identity is corrupt")
    release = Path(target_release)
    for name, expected in identity["source_files"].items():
        if file_record(release / safe_relative(name))["sha256"] != expected:
            raise ValueError(f"Frozen training source changed: {name}")
    if file_record(release / "vendor/prime-rl/uv.lock")["sha256"] != identity["lock_sha256"]:
        raise ValueError("Frozen runtime lock changed")
    return identity


def validate_assets(target, baseline, identity, release, source_assets_receipt, source_data_manifest):
    planned = target.model_dump(mode="json")
    available = baseline.model_dump(mode="json")
    if any(planned[key] != available[key] for key in planned if key not in PATH_FIELDS):
        raise ValueError("Target asset baseline differs from the planned scientific configuration")
    original = read_json(source_assets_receipt)
    model = read_json(Path(release) / "manifests/model.json")
    if (model["repo_id"], model["revision"]) != (original["model_id"], original["revision"]):
        raise ValueError("Migration changed the pinned model")
    records = {item["name"]: {"size": item["size"], "sha256": item["sha256"]} for item in model["files"]}
    source_records = {item["name"]: {"size": item["size"], "sha256": item["sha256"]} for item in original["files"]}
    if records != source_records:
        raise ValueError("Original model receipt and frozen model manifest disagree")
    for name, expected in records.items():
        if file_record(baseline.model_path / safe_relative(name)) != expected:
            raise ValueError(f"Target model asset checksum mismatch: {name}")
    dataset = file_record(baseline.dataset_path)
    if dataset["sha256"] != original["dataset_sha256"]:
        raise ValueError("Target dataset changed")
    manifest_record = file_record(source_data_manifest)
    if file_record(baseline.data_manifest) != manifest_record:
        raise ValueError("Prepared question manifest bytes changed")
    manifest = read_json(baseline.data_manifest)
    if manifest.get("sha256") != object_digest({k: v for k, v in manifest.items() if k != "sha256"}):
        raise ValueError("Prepared question manifest integrity failed")
    if manifest["sha256"] != identity["data_sha256"] or manifest["contract"]["source_sha256"] != dataset["sha256"]:
        raise ValueError("Prepared questions differ from the original run identity")
    prepared = read_json(baseline.prepared_model_path / "study-assets.json")
    for key in ("model_id", "revision", "dataset_sha256", "prepared_tokenizer_config_sha256", "tokenizer_parity"):
        if prepared[key] != original[key]:
            raise ValueError(f"Prepared native tokenizer identity changed: {key}")
    native = file_record(baseline.prepared_model_path / "tokenizer_config.json")
    if native["sha256"] != original["prepared_tokenizer_config_sha256"]:
        raise ValueError("Prepared native tokenizer metadata changed")
    for name, expected in records.items():
        if name == "tokenizer_config.json":
            continue
        prepared_file = baseline.prepared_model_path / safe_relative(name)
        if prepared_file.is_symlink():
            if prepared_file.resolve() != (baseline.model_path / name).resolve():
                raise ValueError("Prepared native model symlink references another asset")
        elif file_record(prepared_file) != expected:
            raise ValueError("Prepared native model file differs from the pinned asset")
    return {
        "model": records,
        "dataset": dataset,
        "data_manifest": manifest_record,
        "native_tokenizer_config": native,
        "original_assets_receipt": original,
    }


def validate_checkpoint(source, study, identity, files):
    marker = read_json(source / "study/complete.json")
    backup = read_json(source / "backup-verified.json")
    if backup.get("checkpoint") != marker:
        raise ValueError("Original shared backup marker differs from its verified receipt")
    expected_files = {
        key: {"size": value["bytes"], "sha256": value["sha256"]} for key, value in backup["files"].items()
    }
    if {key: value for key, value in files.items() if key != "backup-verified.json"} != expected_files:
        raise ValueError("Original shared backup failed its all-file checksum verification")
    if (
        marker.get("format") != 2
        or marker.get("config_sha256") != study.fingerprint()
        or marker.get("lag") != study.lag
    ):
        raise ValueError("Original checkpoint configuration differs from the supplied source study")
    if marker.get("identity_sha256") != identity["sha256"]:
        raise ValueError("Original checkpoint source/runtime identity differs")
    if source.name != f"step_{marker['step']}" or not study.lag < marker["step"] < study.max_steps:
        raise ValueError("Migration requires an unfinished checkpoint after bootstrap")
    for name, key in (("components.json", "components_sha256"), ("queue.pkl", "queue_sha256")):
        if files[f"study/{name}"]["sha256"] != marker[key]:
            raise ValueError(f"Original checkpoint {name} checksum mismatch")
    components = read_json(source / "study/components.json")
    required = {"trainer/.metadata", "orchestrator/progress.pt", *(f"rng/rank_{rank}.pt" for rank in range(4))}
    component_paths = [str(safe_relative(item["path"])) for item in components]
    if len(component_paths) != len(set(component_paths)) or not required.issubset(component_paths):
        raise ValueError("Missing or duplicated full recovery checkpoint components")
    if {name for name in component_paths if name.startswith("rng/")} != {f"rng/rank_{rank}.pt" for rank in range(4)}:
        raise ValueError("Checkpoint trainer RNG rank layout differs")
    if len([name for name in component_paths if name.startswith("trainer/") and name.endswith(".distcp")]) < 4:
        raise ValueError("Checkpoint is missing distributed trainer state")
    for item in components:
        actual = files.get(item["path"])
        if (
            actual is None
            or actual["size"] != item["size"]
            or ("sha256" in item and actual["sha256"] != item["sha256"])
        ):
            raise ValueError("Full recovery checkpoint component checksum mismatch")
    with (source / "study/queue.pkl").open("rb") as stream:
        state = pickle.load(stream)
    if not isinstance(state, AsyncQueueState):
        raise ValueError("Migration requires an asynchronous queue checkpoint")
    state.validate()
    if state.jobs:
        raise ValueError("Migration refuses pending historical generation jobs")
    if (state.completed_steps, state.lag, state.generation_stop) != (
        marker["step"],
        study.lag,
        study.max_steps - study.lag,
    ):
        raise ValueError("Checkpoint queue policy clock or generation horizon differs")
    for cohort in state.pending.values():
        if len(cohort.response_ids) != study.response_batch_size:
            raise ValueError("Checkpoint cohort response count differs")
    return (
        marker,
        backup,
        {
            "completed_steps": state.completed_steps,
            "lag": state.lag,
            "ready_versions": sorted(state.pending),
            "ready_cohorts": len(state.pending),
            "pending_jobs": len(state.jobs),
            "generation_stop": state.generation_stop,
            "generated_cohorts": state.generated_cohorts,
        },
    )


def atomic_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".migration-receipt-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.unlink(temporary)


def migrate(
    *,
    source_checkpoint,
    destination_checkpoint,
    source_study,
    target_study,
    source_identity,
    target_release,
    target_baseline,
    source_assets_receipt,
    source_data_manifest,
    receipt_path,
    target_checkpoint,
    allow_local_retention_change=False,
):
    source, destination = Path(source_checkpoint), Path(destination_checkpoint)
    receipt_path, local_target = Path(receipt_path), Path(target_checkpoint)
    if not all(path.is_absolute() for path in (source, destination, receipt_path, local_target)):
        raise ValueError("Checkpoint and receipt paths must be absolute")
    local_target = local_target.resolve()
    if destination.exists() or receipt_path.exists() or destination.is_symlink() or receipt_path.is_symlink():
        raise FileExistsError("Migration cannot overwrite a checkpoint or receipt")
    roots = [source.resolve(), destination.resolve(), receipt_path.resolve()]
    if any(a.is_relative_to(b) or b.is_relative_to(a) for index, a in enumerate(roots) for b in roots[index + 1 :]):
        raise ValueError("Original checkpoint, adapted checkpoint and audit receipt must be separate")
    source_config, target_config = StudyConfig.read(source_study), StudyConfig.read(target_study)
    baseline = StudyConfig.read(target_baseline)
    workspace = local_target.parent.parent
    expected_paths = {
        "model_path": workspace / "assets/model",
        "dataset_path": workspace / "assets/train.parquet",
        "data_manifest": workspace / "assets/train-manifest.json",
        "prepared_model_path": workspace / "assets/native-model",
        "output_dir": workspace / "runs" / target_config.output_dir.name,
        "metrics_mirror_root": None,
        "historical_rollouts": baseline.historical_rollouts,
    }
    if local_target.parent.name != "resume" or any(
        (getattr(target_config, key).resolve() if getattr(target_config, key) is not None else None)
        != (value.resolve() if value is not None else None)
        for key, value in expected_paths.items()
    ):
        raise ValueError("Target study differs from the planned node-local storage relocation")
    changes = validate_config_change(
        source_config,
        target_config,
        allow_local_retention_change=allow_local_retention_change,
    )
    identity = validate_source_identity(source_identity, target_release)
    assets = validate_assets(
        target_config, baseline, identity, target_release, source_assets_receipt, source_data_manifest
    )
    source_files = inventory(source)
    marker, backup, queue = validate_checkpoint(source, source_config, identity, source_files)
    if destination.name != source.name or local_target.name != source.name:
        raise ValueError("Adapted checkpoint paths must preserve the policy step")
    replacement = {**marker, "config_sha256": target_config.fingerprint()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".migration-"))
    try:
        for name, expected in source_files.items():
            target = staging / safe_relative(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with (source / name).open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if file_record(target) != expected:
                raise ValueError(f"Checkpoint copy checksum mismatch: {name}")
        with (staging / "study/complete.json").open("w") as stream:
            json.dump(replacement, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        copied_files = inventory(staging)
        if {key for key in source_files if source_files[key] != copied_files[key]} != {"study/complete.json"}:
            raise ValueError("Migration modified checkpoint payload bytes")
        if inventory(source) != source_files:
            raise ValueError("Original checkpoint changed during migration")
        receipt = {
            "format": 1,
            "status": "verified",
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "target_checkpoint": str(local_target),
            "adapted_checkpoint": str(destination),
            "source_checkpoint": str(source),
            "checkpoint_step": marker["step"],
            "config_sha256": replacement["config_sha256"],
            "identity_sha256": marker["identity_sha256"],
            "complete_sha256": copied_files["study/complete.json"]["sha256"],
            "components_sha256": marker["components_sha256"],
            "queue_sha256": marker["queue_sha256"],
            "files": copied_files,
            "source_files": source_files,
            "source_complete": marker,
            "source_backup_receipt": backup,
            "source_identity": identity,
            "before_config": source_config.model_dump(mode="json"),
            "after_config": target_config.model_dump(mode="json"),
            "config_diff": changes,
            "local_retention_change": {
                "explicitly_enabled": allow_local_retention_change,
                "before_keep_interval": source_config.checkpoint_keep_interval,
                "after_keep_interval": target_config.checkpoint_keep_interval,
                "checkpoint_interval": target_config.checkpoint_interval,
                "before_keep_last": source_config.checkpoint_keep_last,
                "after_keep_last": target_config.checkpoint_keep_last,
                "shared_retention_modified": False,
                "requires_shared_milestone_verification_before_local_pruning": allow_local_retention_change,
            },
            "queue": queue,
            "assets": assets,
            "runtime_identity_required_at_launch": True,
        }
        for directory in [*sorted((path for path in staging.rglob("*") if path.is_dir()), reverse=True), staging]:
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Adapted checkpoint destination appeared during migration")
        os.rename(staging, destination)
        atomic_json_new(receipt_path, receipt)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-local-retention-change", action="store_true")
    for name in (
        "source-checkpoint",
        "destination-checkpoint",
        "source-study",
        "target-study",
        "source-identity",
        "target-release",
        "target-baseline",
        "source-assets-receipt",
        "source-data-manifest",
        "receipt-path",
        "target-checkpoint",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    result = migrate(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("status", "checkpoint_step", "config_sha256", "queue")}))


if __name__ == "__main__":
    main()
