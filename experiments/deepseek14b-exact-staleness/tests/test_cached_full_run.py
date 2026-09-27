import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("prepare_cached_full_run", SCRIPTS / "prepare_cached_full_run.py")
CACHED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHED)


def specification(tmp_path):
    return {
        "source": str(tmp_path / "source"),
        "old_profile": str(tmp_path / "old-profile"),
        "root": str(tmp_path / "study"),
        "output": str(tmp_path / "study/outputs/fresh"),
        "model_id": "Qwen/Qwen2.5-3B",
        "model_revision": "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
        "allowed_nodes": ["deep-chungus-7"],
        "allocated_gpus": 8,
        "min_gpu_bytes": 79_000_000_000,
        "allow_seven_healthy": True,
    }


def probes(failed=()):
    return {
        "allocated_devices": list(map(str, range(8))),
        "healthy_devices": [str(index) for index in range(8) if index not in failed],
        "results": [
            {
                "device": str(index),
                "healthy": index not in failed,
                "uuid": f"uuid-{index}",
                "name": "NVIDIA A100-SXM4-80GB",
                "bytes": 85_100_000_000,
            }
            for index in range(8)
        ],
    }


def test_prelaunch_inventory_rejects_tampered_files_before_preparation(tmp_path):
    for name in CACHED.CONTROL_FILES:
        (tmp_path / name).write_text("script")
    (tmp_path / "preparation-input.json").write_text(json.dumps(specification(tmp_path)))
    inventory = {
        "schema_version": 1,
        "files_sha256": {name: CACHED.digest(tmp_path / name) for name in CACHED.CONTROL_FILES},
    }
    (tmp_path / "prelaunch.json").write_text(json.dumps(inventory))
    assert CACHED.verify_prelaunch(tmp_path) == specification(tmp_path)
    (tmp_path / "launch_full_run.py").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        CACHED.verify_prelaunch(tmp_path)


@pytest.mark.parametrize("alteration", ["missing", "outside"])
def test_prelaunch_inventory_requires_exact_authorized_file_set(tmp_path, alteration):
    inventory = {name: "hash" for name in CACHED.CONTROL_FILES}
    if alteration == "missing":
        inventory.pop("cached_full_run_job.sh")
    else:
        inventory["../unrelated"] = "hash"
    (tmp_path / "prelaunch.json").write_text(json.dumps({"schema_version": 1, "files_sha256": inventory}))
    with pytest.raises(ValueError, match="exactly"):
        CACHED.verify_prelaunch(tmp_path)


@pytest.mark.parametrize(
    "changes", [{"SLURMD_NODENAME": "deep-chungus-5"}, {"SLURM_RESTART_COUNT": "1"}, {"SLURM_JOB_ID": ""}]
)
def test_cached_preparation_requires_fresh_authorized_allocation(tmp_path, changes):
    environment = {
        "SLURMD_NODENAME": "deep-chungus-7",
        "SLURM_JOB_ID": "123",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    }
    with pytest.raises(ValueError, match="Unexpected node"):
        CACHED.validate_specification(specification(tmp_path), {**environment, **changes})
    assert CACHED.validate_specification(specification(tmp_path), environment) == (
        "deep-chungus-7",
        list(map(str, range(8))),
    )


@pytest.mark.parametrize("failed,inference", [((), 4), ((7,), 3)])
def test_cached_preparation_uses_every_healthy_gpu(tmp_path, failed, inference):
    probe = probes(failed)
    visible, inferred = CACHED.choose_topology(probe, specification(tmp_path), probe["allocated_devices"])
    assert inferred == inference
    assert visible == probe["healthy_devices"]


def test_seven_gpu_fallback_requires_authorization_and_never_hides_occupation(tmp_path):
    probe = probes((7,))
    values = {**specification(tmp_path), "allow_seven_healthy": False}
    with pytest.raises(ValueError, match="topology"):
        CACHED.choose_topology(probe, values, probe["allocated_devices"])
    probe["results"][7]["error"] = "GPU is already occupied: another process"
    with pytest.raises(ValueError, match="occupied"):
        CACHED.choose_topology(probe, specification(tmp_path), probe["allocated_devices"])


@pytest.mark.parametrize("change", ["memory", "identity", "outside", "too_few"])
def test_cached_preparation_rejects_incompatible_devices(tmp_path, change):
    probe = probes((6, 7) if change == "too_few" else ())
    if change == "memory":
        probe["results"][0]["bytes"] = 40_000_000_000
    elif change == "identity":
        probe["results"][0]["uuid"] = probe["results"][1]["uuid"]
    elif change == "outside":
        probe["healthy_devices"][0] = "outside"
    with pytest.raises(ValueError):
        CACHED.choose_topology(probe, specification(tmp_path), probe["allocated_devices"])


def save_manifest(path, identifiers):
    body = {
        "format": 1,
        "source_rows": len(identifiers),
        "included_rows": len(identifiers),
        "excluded_rows": 0,
        "records": [{"id": identifier, "included": True} for identifier in identifiers],
        "contract": {
            "source_sha256": "dataset",
            "reward": "grader-v3",
            "prompt_instruction": "boxed",
            "prompt_max_tokens": 2048,
            "reward_timeout_seconds": 8,
        },
    }
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path.write_text(json.dumps({**body, "sha256": digest}))


def test_cached_preparation_checks_ordered_question_membership(tmp_path):
    reference, prepared = tmp_path / "reference.json", tmp_path / "prepared.json"
    save_manifest(reference, ["a", "b"])
    save_manifest(prepared, ["a", "b"])
    assert CACHED.verify_membership(reference, prepared) == 2
    save_manifest(prepared, ["b", "a"])
    with pytest.raises(ValueError, match="membership or order"):
        CACHED.verify_membership(reference, prepared)
    save_manifest(prepared, ["a", "different"])
    with pytest.raises(ValueError, match="membership or order"):
        CACHED.verify_membership(reference, prepared)


def test_release_freeze_excludes_vendor_and_bytecode(tmp_path):
    for name in (
        "src/study.py",
        "scripts/launch.py",
        "scripts/job.sh",
        "pyproject.toml",
        "manifests/model.json",
        "src/__pycache__/study.pyc",
        "vendor/prime-rl/sensitive-file",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    files = CACHED.freeze_release(tmp_path)
    assert set(files) == {
        "src/study.py",
        "scripts/launch.py",
        "scripts/job.sh",
        "pyproject.toml",
        "manifests/model.json",
    }
    assert not any("vendor" in name or "pycache" in name for name in files)
    with pytest.raises(FileExistsError):
        CACHED.freeze_release(tmp_path)


def test_cached_preparation_passes_cold_start_deadline_to_child_probe(tmp_path, monkeypatch):
    import launch_full_run

    control = tmp_path / "control"
    control.mkdir()
    verified = []
    warmed = []

    def verify(control):
        verified.append(control)
        return specification(tmp_path)

    class ProbeInspected(Exception):
        pass

    def inspect_stage(name, command, environment, timeout, cwd):
        assert verified == [control]
        assert warmed == [control / "import-warmup-preparation.json"]
        assert name == "probe_all_allocated_gpus"
        assert command[command.index("--timeout") + 1] == "300"
        assert timeout == 360
        assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        raise ProbeInspected

    monkeypatch.setattr(CACHED, "verify_prelaunch", verify)
    monkeypatch.setattr(launch_full_run, "warm_torch_import", lambda path, environment: warmed.append(path))
    monkeypatch.setattr(CACHED, "run_stage", inspect_stage)
    monkeypatch.setenv("SLURMD_NODENAME", "deep-chungus-7")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    with pytest.raises(ProbeInspected):
        CACHED.prepare(control)


def test_failed_import_warmup_never_starts_parallel_gpu_probes(tmp_path, monkeypatch):
    import launch_full_run

    control = tmp_path / "control"
    control.mkdir()
    stages = []

    def fail_import(path, environment):
        assert path == control / "import-warmup-preparation.json"
        raise RuntimeError("Torch import warmup failed")

    monkeypatch.setattr(CACHED, "verify_prelaunch", lambda path: specification(tmp_path))
    monkeypatch.setattr(launch_full_run, "warm_torch_import", fail_import)
    monkeypatch.setattr(CACHED, "run_stage", lambda *args: stages.append(args))
    monkeypatch.setenv("SLURMD_NODENAME", "deep-chungus-7")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    with pytest.raises(RuntimeError, match="Torch import warmup failed"):
        CACHED.prepare(control)
    assert stages == []
    assert not (control / "work").exists()
