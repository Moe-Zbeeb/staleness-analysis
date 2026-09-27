import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load("prepare_full_run")
LAUNCH = load("launch_full_run")


def probes(count, failed=()):
    return {
        "allocated_devices": [str(i) for i in range(count)],
        "healthy_devices": [str(i) for i in range(count) if i not in failed],
        "results": [{"device": str(i), "healthy": i not in failed, "uuid": f"uuid-{i}"} for i in range(count)],
    }


def test_full_run_preserves_science_and_budget(tmp_path):
    baseline = json.loads((SCRIPTS.parent / "configs/exact256-80gb-seed42-v2.json").read_text())
    result = PREPARE.full_config(baseline, tmp_path / "full", 5)
    assert {k for k in baseline if baseline[k] != result[k]} == {"output_dir", "inference_gpus"}
    assert result["max_steps"] == 1000 and result["lag"] == 256 and result["checkpoint_interval"] == 100


@pytest.mark.parametrize("key,value", [("max_steps", 4), ("lag", 32), ("weight_decay", 0.01)])
def test_full_run_rejects_wrong_protocol(key, value, tmp_path):
    baseline = json.loads((SCRIPTS.parent / "configs/exact256-80gb-seed42-v2.json").read_text())
    baseline[key] = value
    with pytest.raises(ValueError, match="protocol"):
        PREPARE.full_config(baseline, tmp_path / "full", 5)


def test_all_nine_gpus_and_authorized_seven_gpu_fallback():
    assert LAUNCH.select_devices(probes(8), 8, 8) == [str(i) for i in range(8)]
    assert LAUNCH.select_devices(probes(9), 9, 9) == [str(i) for i in range(9)]
    assert LAUNCH.select_devices(probes(8, {7}), 8, 7) == [str(i) for i in range(7)]


def test_occupied_gpu_is_not_a_hardware_fallback():
    value = probes(8, {7})
    value["results"][7]["error"] = "GPU is already occupied: 10% free"
    with pytest.raises(ValueError, match="occupied"):
        LAUNCH.select_devices(value, 8, 7)


def test_full_run_rejects_duplicate_gpu_aliases():
    value = probes(9)
    value["results"][8]["uuid"] = value["results"][7]["uuid"]
    with pytest.raises(ValueError, match="duplicated"):
        LAUNCH.select_devices(value, 9, 9)


def test_full_run_never_silently_discards_healthy_gpus():
    with pytest.raises(ValueError, match="topology"):
        LAUNCH.select_devices(probes(9), 9, 8)


@pytest.mark.parametrize("inference_gpus", [3, 4])
def test_fresh_study_uses_full_eight_gpu_allocation_with_only_authorized_fallback(tmp_path, inference_gpus):
    baseline = json.loads((SCRIPTS.parent / "configs/exact256-80gb-seed42-v2.json").read_text())
    result = PREPARE.fresh_config(baseline, tmp_path / "full", inference_gpus, 8, 79_000_000_000)
    assert result["trainer_gpus"] == 4
    assert result["inference_gpus"] == inference_gpus
    assert result["max_steps"] == 1000 and result["lag"] == 256


@pytest.mark.parametrize(
    "inference_gpus,allocated_gpus,minimum_bytes",
    [(5, 9, 79_000_000_000), (3, 7, 79_000_000_000), (4, 8, 39_000_000_000)],
)
def test_fresh_study_rejects_unrequested_hardware_layout(tmp_path, inference_gpus, allocated_gpus, minimum_bytes):
    baseline = json.loads((SCRIPTS.parent / "configs/exact256-80gb-seed42-v2.json").read_text())
    with pytest.raises(ValueError, match="exclusive eight-A100-80GB"):
        PREPARE.fresh_config(baseline, tmp_path / "full", inference_gpus, allocated_gpus, minimum_bytes)


def test_prepared_nodes_accept_repeat_or_comma_but_not_slurm_expressions():
    assert PREPARE.allowed_nodes(["deep-chungus-9,deep-chungus-10", "deep-chungus-11"]) == [
        "deep-chungus-9",
        "deep-chungus-10",
        "deep-chungus-11",
    ]
    with pytest.raises(ValueError, match="explicit candidate"):
        PREPARE.allowed_nodes(["deep-chungus-[9-11]"])


def test_launcher_accepts_only_prepared_nodes_and_never_an_automatic_restart():
    manifest = {"allowed_nodes": ["deep-chungus-9", "deep-chungus-11"]}
    assert LAUNCH.validate_node(manifest, {"SLURMD_NODENAME": "deep-chungus-11"}) == "deep-chungus-11"
    assert LAUNCH.validate_node({"node": "deep-chungus-7"}, {"SLURMD_NODENAME": "deep-chungus-7"}) == "deep-chungus-7"
    for environment in (
        {"SLURMD_NODENAME": "deep-chungus-10"},
        {"SLURMD_NODENAME": "deep-chungus-9", "SLURM_RESTART_COUNT": "1"},
    ):
        with pytest.raises(ValueError, match="Unexpected node or unsafe automatic restart"):
            LAUNCH.validate_node(manifest, environment)
    with pytest.raises(ValueError, match="Invalid prepared node allowlist"):
        LAUNCH.validate_node({"allowed_nodes": "deep-chungus-9"}, {"SLURMD_NODENAME": "deep"})


def test_control_accepts_preparation_artifacts_but_not_existing_run_state(tmp_path):
    for name in (
        "launch_full_run.py",
        "full_run_job.sh",
        "probe_allocated_gpus.py",
        "prepare_cached_full_run.py",
        "prelaunch.json",
    ):
        (tmp_path / name).write_text("frozen")
    (tmp_path / "work").mkdir()
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "prepare_small_model_profile.cpython-312.pyc").write_bytes(b"python import cache")
    (tmp_path / "device-probes-preparation.json").write_text("receipt")
    hashes = PREPARE.control_scripts(tmp_path)
    assert "prepare_cached_full_run.py" in hashes and "prelaunch.json" in hashes
    assert "device-probes-preparation.json" not in hashes
    (tmp_path / "study.json").write_text("old")
    with pytest.raises(FileExistsError, match="existing run"):
        PREPARE.control_scripts(tmp_path)


@pytest.mark.parametrize("invalid", [None, "format", "pin"])
def test_fresh_preparation_validates_assets_and_identity_without_a_profile(study, tmp_path, monkeypatch, invalid):
    import deepseek_study
    from deepseek_study.dataset import assets
    from deepseek_study.runtime import identity

    release = tmp_path / "release"
    release.mkdir()
    (release / "pyproject.toml").write_text("locked")
    (release / "PACKAGE_SHA256.json").write_text(
        json.dumps({"pyproject.toml": PREPARE.digest(release / "pyproject.toml")})
    )
    control = tmp_path / "control"
    control.mkdir()
    for name in ("launch_full_run.py", "full_run_job.sh", "probe_allocated_gpus.py"):
        (control / name).write_text("frozen launcher")
    config = study.model_copy(
        update={
            "max_steps": 1000,
            "lag": 256,
            "checkpoint_interval": 100,
            "trainer_gpus": 4,
            "inference_gpus": 4,
            "inference_tensor_parallel": 1,
            "reasoning_required": invalid == "format",
        }
    )
    source = tmp_path / "input.json"
    source.write_text(config.model_dump_json())
    model = "Qwen/Qwen2.5-3B"
    revision = "wrong" if invalid == "pin" else PREPARE.PROFILE_MODELS[model][0]
    monkeypatch.setattr(deepseek_study, "MODEL_ID", model)
    monkeypatch.setattr(deepseek_study, "MODEL_REVISION", revision)
    monkeypatch.setattr(assets, "MODEL_ID", model)
    validated = []

    def validate(value):
        assets.validate_grading_format(value)
        validated.append(value)
        return {"rows": 37696}

    monkeypatch.setattr(assets, "validate_prepared", validate)
    monkeypatch.setattr(identity, "capture", lambda path, value: {"sha256": "fresh-source-data-runtime"})
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_full_run.py",
            "--study",
            str(source),
            "--release",
            str(release),
            "--control",
            str(control),
            "--output",
            str(config.output_dir),
            "--inference-gpus",
            "4",
            "--allocated-gpus",
            "8",
            "--node",
            "deep-chungus-9,deep-chungus-11",
            "--minimum-gpu-bytes",
            "79000000000",
        ],
    )
    if invalid:
        with pytest.raises(ValueError, match="pinned"):
            PREPARE.main()
        assert not (control / "full-run.json").exists()
        assert not (control / "study.json").exists()
    else:
        PREPARE.main()
        manifest = json.loads((control / "full-run.json").read_text())
        assert len(validated) == 1
        assert manifest["preparation_mode"] == "fresh_study" and manifest["profile"] is None
        assert manifest["identity_sha256"] == "fresh-source-data-runtime"
        assert manifest["asset_preflight"] == {"rows": 37696}
        assert manifest["allowed_nodes"] == ["deep-chungus-9", "deep-chungus-11"]
        assert manifest["maximum_updates"] == 1000 and manifest["lag"] == 256
        assert manifest["startup_deadlines"] == LAUNCH.startup_deadlines()
        assert "node" not in manifest
        assert not config.output_dir.exists()


def test_verified_profile_preparation_remains_supported(study, tmp_path, monkeypatch):
    import deepseek_study
    from deepseek_study.runtime import identity

    profile = tmp_path / "profile"
    release = profile / "release"
    release.mkdir(parents=True)
    (release / "pyproject.toml").write_text("frozen")
    (release / "PACKAGE_SHA256.json").write_text(
        json.dumps({"pyproject.toml": PREPARE.digest(release / "pyproject.toml")})
    )
    model = "Qwen/Qwen3-1.7B"
    revision = PREPARE.PROFILE_MODELS[model][0]
    (profile / "preparation.json").write_text(
        json.dumps(
            {
                "model_id": model,
                "model_revision": revision,
                "same_question_membership_and_order": True,
                "prime_rl_modified": False,
            }
        )
    )
    baseline = study.model_copy(update={"max_steps": 1000, "lag": 256, "checkpoint_interval": 100, "trainer_gpus": 4})
    (profile / "study.json").write_text(baseline.model_dump_json())
    control = tmp_path / "control"
    control.mkdir()
    for name in ("launch_full_run.py", "full_run_job.sh", "probe_allocated_gpus.py"):
        (control / name).write_text("frozen launcher")
    monkeypatch.setattr(deepseek_study, "MODEL_ID", model)
    monkeypatch.setattr(deepseek_study, "MODEL_REVISION", revision)
    monkeypatch.setattr(identity, "capture", lambda path, value: {"sha256": "profile-source-data-runtime"})
    requested_identities = []

    def read_identity(path):
        requested_identities.append(path)
        return {"sha256": "profile-source-data-runtime"}

    monkeypatch.setattr(identity, "read_identity", read_identity)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_full_run.py",
            "--profile",
            str(profile),
            "--control",
            str(control),
            "--output",
            str(tmp_path / "new-output"),
            "--inference-gpus",
            "4",
            "--allocated-gpus",
            "8",
            "--node",
            "deep-chungus-11",
            "--minimum-gpu-bytes",
            "79000000000",
        ],
    )
    PREPARE.main()
    manifest = json.loads((control / "full-run.json").read_text())
    assert requested_identities == [baseline.output_dir / "source/identity.json"]
    assert manifest["preparation_mode"] == "verified_profile"
    assert manifest["profile"] == str(profile)
    assert manifest["node"] == "deep-chungus-11"
    assert manifest["allowed_nodes"] == ["deep-chungus-11"]
    assert manifest["asset_preflight"] is None


def test_full_launch_preserves_cold_start_probe_and_collective_deadlines(study, tmp_path, monkeypatch):
    from deepseek_study.runtime import identity

    control = tmp_path / "control"
    control.mkdir()
    release = tmp_path / "release"
    release.mkdir()
    study = study.model_copy(update={"trainer_gpus": 4, "inference_gpus": 4})
    (control / "study.json").write_text(study.model_dump_json())
    manifest = {
        "study_sha256": PREPARE.digest(control / "study.json"), "scripts_sha256": {},
        "node": "deep-chungus-11", "release": str(release), "config_sha256": study.fingerprint(),
        "identity_sha256": "source", "allocated_gpus": 8, "minimum_gpu_bytes": 79_000_000_000,
        "startup_deadlines": LAUNCH.startup_deadlines(),
    }
    (control / "full-run.json").write_text(json.dumps(manifest))
    calls = []
    commands = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "--output" in command:
            assert command[command.index("--timeout") + 1] == "300"
            assert kwargs["timeout"] == 360
            Path(command[command.index("--output") + 1]).write_text(json.dumps(probes(8)))
        else:
            assert "torch.distributed.run" in command
            assert kwargs["timeout"] == 900
            Path(command[command.index("--receipt") + 1]).write_text(json.dumps({
                "world_size": 8, "devices": [{"name": "NVIDIA A100 80GB", "bytes": 85_000_000_000}] * 8,
            }))

    monkeypatch.setattr(identity, "capture", lambda *args: {"sha256": "source"})
    monkeypatch.setattr(LAUNCH.subprocess, "run", run)
    monkeypatch.setattr(LAUNCH.os, "execv", lambda executable, command: commands.append(command))
    monkeypatch.setattr(LAUNCH.os, "chdir", lambda path: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["launch_full_run.py", "--control", str(control)])
    monkeypatch.setenv("SLURMD_NODENAME", "deep-chungus-11")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    LAUNCH.main()
    assert len(calls) == 2
    assert commands == [[sys.executable, "-m", "deepseek_study.cli", "run", str(control / "study.json")]]
    receipt = json.loads((control / "job-123/launch.json").read_text())
    assert receipt["startup_deadlines"] == {
        "gpu_probe_seconds": 300, "gpu_probe_parent_margin_seconds": 60,
        "gpu_probe_parent_seconds": 360, "collective_health_seconds": 900,
    }
