import importlib.util
import json
import os
import shutil
import time
from pathlib import Path

import pytest


@pytest.fixture
def archive(tmp_path, monkeypatch):
    path = Path(__file__).with_name("archive_model_caches_dedup.py")
    spec = importlib.util.spec_from_file_location("dedup_archive", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path / "shared")
    monkeypatch.setattr(module, "SOURCE_ROOT", tmp_path / "local")
    monkeypatch.setattr(module, "REQUIRED_UID", os.getuid())
    monkeypatch.setattr(module, "verify_environment", lambda: None)
    monkeypatch.setattr(module, "check_processes", lambda source=None: None)
    module.ROOT.mkdir()
    module.SOURCE_ROOT.mkdir()
    for job, model in module.EXPECTED.items():
        source = module.source_for(job)
        source.mkdir(parents=True)
        (source / "weights.safetensors").write_bytes(("weights-" + model).encode())
        (source / "config.json").write_text(json.dumps({"model": model}))
        (source.parent / "preserved.log").write_bytes(b"original logs")
        (source.parent / "runtime").mkdir()
    return module


def old_receipt(archive, job, remove_source=False):
    source = archive.source_for(job)
    target = archive.ROOT / source.parent.name / source.name
    shutil.copytree(source, target)
    files, _ = archive.inventory(target, hashes=True)
    receipt = {"source": str(source), "archive": str(target), "files": files, "verified_at": time.time()}
    archive.receipt_for(job).write_text(json.dumps(receipt))
    if remove_source:
        shutil.rmtree(source)
    return target


def test_duplicates_copy_and_verify_canonical_once(archive, monkeypatch):
    copied = []
    original = archive.copy_model

    def tracked(source, target, expected, stamps):
        copied.append(target)
        return original(source, target, expected, stamps)

    monkeypatch.setattr(archive, "copy_model", tracked)
    records = archive.run()
    assert len(records) == 7 and len(copied) == 3
    groups = {}
    for job, model in archive.EXPECTED.items():
        assert not archive.source_for(job).exists()
        assert (archive.source_for(job).parent / "preserved.log").read_bytes() == b"original logs"
        assert (archive.source_for(job).parent / "runtime").is_dir()
        receipt = archive.load_verified(job)
        groups.setdefault(model, set()).add(receipt["archive"])
    assert all(len(targets) == 1 for targets in groups.values())
    assert len(archive.run()) == 7


def test_existing_old_verified_canonical_is_reused_without_rehash(archive, monkeypatch):
    first = next(iter(archive.EXPECTED))
    canonical = old_receipt(archive, first, remove_source=True)
    hashed = []
    original = archive.digest

    def tracked(path):
        hashed.append(Path(path))
        return original(path)

    monkeypatch.setattr(archive, "digest", tracked)
    archive.run()
    assert not any(path.is_relative_to(canonical) for path in hashed)
    duplicate = list(archive.EXPECTED)[1]
    assert archive.load_verified(duplicate)["archive"] == str(canonical)
    assert any(path.is_relative_to(archive.source_for(duplicate)) for path in hashed)


def test_different_source_manifest_gets_separate_verified_archive(archive):
    second = list(archive.EXPECTED)[1]
    (archive.source_for(second) / "config.json").write_text('{"different": true}')
    archive.run()
    first = next(iter(archive.EXPECTED))
    assert archive.load_verified(first)["archive"] != archive.load_verified(second)["archive"]
    assert len(list((archive.ROOT / "canonical").iterdir())) == 4


@pytest.mark.parametrize("corruption", ["missing", "size", "mtime", "receipt", "extra", "symlink"])
def test_corrupt_existing_archive_or_receipt_refuses_reclamation(archive, corruption):
    first = next(iter(archive.EXPECTED))
    target = old_receipt(archive, first)
    weights = target / "weights.safetensors"
    if corruption == "missing":
        weights.unlink()
    elif corruption == "size":
        weights.write_bytes(weights.read_bytes() + b"changed")
    elif corruption == "mtime":
        os.utime(weights, (time.time() + 1, time.time() + 1))
    elif corruption == "receipt":
        value = json.loads(archive.receipt_for(first).read_text())
        value["source"] = "/unrelated/source"
        archive.receipt_for(first).write_text(json.dumps(value))
    elif corruption == "extra":
        (target / "unexpected").write_bytes(b"x")
    elif corruption == "symlink":
        weights.unlink()
        weights.symlink_to(archive.source_for(first) / "weights.safetensors")
    with pytest.raises((ValueError, FileNotFoundError)):
        archive.run()
    assert archive.source_for(first).exists()
    assert all(archive.source_for(job).exists() for job in archive.EXPECTED)


def test_unfinished_copy_preserves_source_and_publishes_no_receipt(archive, monkeypatch):
    first = next(iter(archive.EXPECTED))
    source = archive.source_for(first)

    def fail(source, target, expected, stamps):
        archive.ensure_directory(target, archive.ROOT)
        (target / "weights.safetensors").write_bytes(b"partial")
        raise OSError("simulated storage failure")

    monkeypatch.setattr(archive, "copy_model", fail)
    with pytest.raises(OSError):
        archive.run()
    assert source.exists()
    assert not archive.receipt_for(first).exists()


def test_partial_unverified_old_destination_is_repaired_without_removing_other_data(archive):
    first = next(iter(archive.EXPECTED))
    source = archive.source_for(first)
    target = archive.ROOT / source.parent.name / source.name
    target.mkdir(parents=True)
    (target / "weights.safetensors").write_bytes(b"partial")
    untouched = target.parent / "unrelated-log.txt"
    untouched.write_text("keep")
    archive.run()
    assert archive.load_verified(first)["archive"] == str(target)
    assert untouched.read_text() == "keep"


def test_missing_source_without_receipt_fails_closed(archive):
    first = next(iter(archive.EXPECTED))
    shutil.rmtree(archive.source_for(first))
    with pytest.raises(ValueError, match="Absent original"):
        archive.run()
    assert all(archive.source_for(job).exists() for job in list(archive.EXPECTED)[1:])


def test_new_receipt_manifest_corruption_is_rejected(archive):
    archive.run()
    job = next(iter(archive.EXPECTED))
    path = archive.receipt_for(job)
    value = json.loads(path.read_text())
    value["files"]["weights.safetensors"]["sha256"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="manifest checksum"):
        archive.run()


def test_reused_receipt_cannot_change_canonical_proof(archive):
    archive.run()
    second = list(archive.EXPECTED)[1]
    path = archive.receipt_for(second)
    value = json.loads(path.read_text())
    value["canonical_receipt"] = str(archive.ROOT / "outside/verified.json")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="canonical|Canonical"):
        archive.run()
