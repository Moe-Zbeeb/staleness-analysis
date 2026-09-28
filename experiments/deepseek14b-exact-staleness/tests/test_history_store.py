import errno
import hashlib
import json
import shutil

import pytest

from deepseek_study.rollouts.history_store import (
    canonical,
    freeze_export,
    load_result,
    pin_job,
    publish_job,
    publish_result,
    validate_job,
)


@pytest.fixture
def export(tmp_path):
    source = tmp_path / "live-weights"
    source.mkdir()
    (source / "model-00001-of-00002.safetensors").write_bytes(b"tensor-shard-one")
    (source / "model-00002-of-00002.safetensors").write_bytes(b"tensor-shard-two")
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer1": "model-00001-of-00002.safetensors",
                    "layer2": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (source / ".ready").write_text("ready")
    (source / "config.json").write_text("{}")
    metadata = {"run_uuid": "run-a", "config_sha256": "config-a", "behavior_version": 7, "lag": 256}
    frozen = tmp_path / "frozen"
    job = freeze_export(source, frozen, metadata)
    return source, frozen, job


def rewritten_identity(job):
    body = {key: value for key, value in job.items() if key != "job_id"}
    return {**body, "job_id": hashlib.sha256(canonical(body)).hexdigest()}


def test_frozen_snapshot_ignores_markers_and_does_not_alias_mutable_source(export):
    source, frozen, job = export
    assert len(job["weights"]) == 3
    assert not (frozen / "weights/.ready").exists()
    assert not (frozen / "weights/config.json").exists()
    (source / "model-00001-of-00002.safetensors").write_bytes(b"replaced current weights")
    assert validate_job(frozen, job) == job
    assert (frozen / "weights/model-00001-of-00002.safetensors").read_bytes() == b"tensor-shard-one"


def test_job_identity_includes_run_config_and_version(export, tmp_path):
    source, _, original = export
    ids = {original["job_id"]}
    for index, (key, value) in enumerate(
        (("run_uuid", "run-b"), ("config_sha256", "config-b"), ("behavior_version", 8))
    ):
        result = freeze_export(source, tmp_path / f"frozen-{index}", {**original["metadata"], key: value})
        ids.add(result["job_id"])
    assert len(ids) == 4


def test_publish_is_verified_idempotent_and_contains_marker(export, tmp_path):
    _, frozen, job = export
    shared = tmp_path / "shared"
    published = publish_job(frozen, shared)
    assert published == shared / "jobs" / job["job_id"]
    assert validate_job(published, job) == job
    assert publish_job(frozen, shared) == published
    assert json.loads((published / "job.json").read_text()) == job


def test_job_corruption_is_detected_before_and_after_publication(export, tmp_path):
    _, frozen, job = export
    published = publish_job(frozen, tmp_path / "shared")
    path = published / "weights" / job["weights"][0]["path"]
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        validate_job(published, job)
    with pytest.raises(ValueError, match="checksum"):
        publish_job(frozen, tmp_path / "shared")
    path = frozen / "weights" / job["weights"][0]["path"]
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        publish_job(frozen, tmp_path / "elsewhere")


def test_wrong_run_and_changed_metadata_cannot_be_accepted(export):
    _, frozen, job = export
    other = rewritten_identity({**job, "metadata": {**job["metadata"], "run_uuid": "another-run"}})
    with pytest.raises(ValueError, match="different run"):
        validate_job(frozen, other)
    with pytest.raises(ValueError, match="different run"):
        load_result(frozen, other)
    with pytest.raises(ValueError, match="identity"):
        validate_job(frozen, {**job, "metadata": {**job["metadata"], "lag": 8}})


def test_partial_job_has_no_usable_publication_and_is_not_overwritten(export, tmp_path):
    _, frozen, job = export
    incomplete = tmp_path / "shared/jobs" / job["job_id"]
    incomplete.mkdir(parents=True)
    evidence = incomplete / "partial"
    evidence.write_bytes(b"keep")
    with pytest.raises(FileNotFoundError):
        publish_job(frozen, tmp_path / "shared")
    assert evidence.read_bytes() == b"keep"
    assert not (incomplete / "job.json").exists()


def test_out_of_order_results_stay_attached_to_their_independent_jobs(export, tmp_path):
    source, frozen, first = export
    next_export = tmp_path / "frozen-next"
    second = freeze_export(source, next_export, {**first["metadata"], "behavior_version": 8})
    first_dir = publish_job(frozen, tmp_path / "shared")
    second_dir = publish_job(next_export, tmp_path / "shared")
    publish_result(second_dir, second, b"second rollout", {"output_tokens": 200})
    assert load_result(first_dir, first) is None
    assert load_result(second_dir, second) == b"second rollout"
    publish_result(first_dir, first, b"first rollout", {"output_tokens": 100})
    assert load_result(first_dir, first) == b"first rollout"
    with pytest.raises(ValueError, match="different run"):
        load_result(second_dir, first)


def test_partial_result_is_unavailable_until_last_marker_and_can_finish_idempotently(export):
    _, frozen, job = export
    (frozen / "result.msgpack").write_bytes(b"complete bytes without marker")
    assert load_result(frozen, job) is None
    result = publish_result(frozen, job, b"complete bytes without marker", {"responses": 512})
    assert load_result(frozen, job) == b"complete bytes without marker"
    assert publish_result(frozen, job, b"complete bytes without marker", {"responses": 512}) == result


def test_conflicting_result_or_metrics_never_overwrites_original(export):
    _, frozen, job = export
    publish_result(frozen, job, b"original", {"responses": 512})
    for archive, metrics in ((b"different", {"responses": 512}), (b"original", {"responses": 16})):
        with pytest.raises(FileExistsError, match="different result"):
            publish_result(frozen, job, archive, metrics)
    assert load_result(frozen, job) == b"original"


def test_conflicting_unmarked_result_is_preserved(export):
    _, frozen, job = export
    (frozen / "result.msgpack").write_bytes(b"partial-or-conflicting")
    with pytest.raises(FileExistsError, match="incomplete"):
        publish_result(frozen, job, b"different", {})
    assert (frozen / "result.msgpack").read_bytes() == b"partial-or-conflicting"
    assert not (frozen / "result.json").exists()


def test_result_checksum_and_metadata_corruption_are_detected(export):
    _, frozen, job = export
    publish_result(frozen, job, b"original", {})
    (frozen / "result.msgpack").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_result(frozen, job)
    (frozen / "result.msgpack").write_bytes(b"original")
    record = json.loads((frozen / "result.json").read_text())
    (frozen / "result.json").write_text(json.dumps({**record, "run_uuid": "other"}))
    with pytest.raises(ValueError, match="different run"):
        load_result(frozen, job)


def test_accepted_result_remains_readable_after_weights_are_cleaned(export):
    _, frozen, job = export
    publish_result(frozen, job, b"saved rollout", {})
    shutil.rmtree(frozen / "weights")
    assert load_result(frozen, job) == b"saved rollout"


def test_failure_receipt_is_fatal_and_prevents_publication(export):
    _, frozen, job = export
    (frozen / "failure.json").write_text(json.dumps({"job_id": job["job_id"], "error": "historical worker failed"}))
    with pytest.raises(RuntimeError, match="historical worker failed"):
        load_result(frozen, job)
    with pytest.raises(RuntimeError, match="failure receipt"):
        publish_result(frozen, job, b"rollout", {})


def test_checkpoint_pin_is_a_complete_independent_snapshot(export, tmp_path):
    _, frozen, job = export
    pinned = pin_job(frozen, tmp_path / "checkpoint/study/history")
    assert validate_job(pinned, job) == job
    assert pin_job(frozen, pinned) == pinned
    for row in job["weights"]:
        assert (frozen / "weights" / row["path"]).stat().st_ino == (pinned / "weights" / row["path"]).stat().st_ino
    shutil.rmtree(frozen)
    assert validate_job(pinned, job) == job


@pytest.mark.parametrize(
    "name", ["../escape.safetensors", "/tmp/escape.safetensors", "a/../../escape.safetensors", "a\\bad.safetensors"]
)
def test_manifest_path_traversal_is_rejected_even_with_a_matching_job_hash(export, name):
    _, frozen, job = export
    malicious = rewritten_identity({**job, "weights": [{**job["weights"][0], "path": name}]})
    with pytest.raises(ValueError, match="path"):
        validate_job(frozen, malicious)


@pytest.mark.parametrize("location", ["source", "manifest_weight", "result", "sharedroot"])
def test_symlinks_are_rejected(export, tmp_path, location):
    source, frozen, job = export
    target = tmp_path / "external"
    target.write_bytes(b"external")
    if location == "source":
        (source / "alias.safetensors").symlink_to(target)
        with pytest.raises(ValueError, match="[Ss]ymlink"):
            freeze_export(source, tmp_path / "another", job["metadata"])
    elif location == "manifest_weight":
        path = frozen / "weights" / job["weights"][0]["path"]
        path.unlink()
        path.symlink_to(target)
        with pytest.raises(ValueError, match="[Ss]ymlink"):
            validate_job(frozen, job)
    elif location == "result":
        (frozen / "result.msgpack").symlink_to(target)
        with pytest.raises(ValueError, match="[Ss]ymlink"):
            publish_result(frozen, job, b"external", {})
    else:
        actual = tmp_path / "real-shared"
        actual.mkdir()
        linked = tmp_path / "shared"
        linked.symlink_to(actual, target_is_directory=True)
        with pytest.raises(ValueError, match="[Ss]ymlink"):
            publish_job(frozen, linked)


def test_index_must_name_existing_local_safetensors(export, tmp_path):
    source, _, job = export
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer1": "absent.safetensors"}}))
    with pytest.raises(ValueError, match="absent"):
        freeze_export(source, tmp_path / "invalid", job["metadata"])
    assert not (tmp_path / "invalid/job.json").exists()


@pytest.mark.parametrize(
    "metadata", [{}, {"run_uuid": "a"}, {"config_sha256": "b"}, {"run_uuid": "", "config_sha256": "b"}]
)
def test_run_and_scientific_config_identity_are_required(export, tmp_path, metadata):
    source, _, _ = export
    with pytest.raises(ValueError, match="identity"):
        freeze_export(source, tmp_path / "invalid", metadata)


def test_checkpoint_pin_falls_back_to_a_verified_copy_across_filesystems(export, tmp_path, monkeypatch):
    from deepseek_study.rollouts import history_store

    _, frozen, job = export

    def cross_device(*args, **kwargs):
        raise OSError(errno.EXDEV, "different filesystems")

    monkeypatch.setattr(history_store.os, "link", cross_device)
    pinned = pin_job(frozen, tmp_path / "checkpoint")
    assert validate_job(pinned, job) == job
    for row in job["weights"]:
        assert (frozen / "weights" / row["path"]).stat().st_ino != (pinned / "weights" / row["path"]).stat().st_ino
