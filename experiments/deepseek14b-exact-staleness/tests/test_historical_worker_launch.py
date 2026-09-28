import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BACKUP = load_script("local_backup")
load_script("launch_full_run")
LOCAL = load_script("node_local_run")
WORKER = load_script("launch_historical_worker")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def allocation(count):
    return {
        "SLURM_JOB_ID": "1234",
        "SLURMD_NODENAME": "deep-worker",
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(count)),
    }


def probes_for(environment, memory=40_000_000_000):
    devices = environment["CUDA_VISIBLE_DEVICES"].split(",")
    return {
        "job_id": environment["SLURM_JOB_ID"],
        "node": environment["SLURMD_NODENAME"],
        "allocated_devices": devices,
        "healthy_devices": devices.copy(),
        "results": [
            {
                "device": device,
                "healthy": True,
                "name": "NVIDIA A100-SXM4-40GB",
                "uuid": "GPU-" + device,
                "bytes": memory,
                "initial_total_bytes": memory,
                "initial_free_bytes": memory,
                "bf16_backward": True,
            }
            for device in devices
        ],
    }


@pytest.fixture
def prepared(tmp_path):
    control, release, workspace = [tmp_path / name for name in ("control", "release", "workspace")]
    for path in (control, release, workspace):
        path.mkdir()
    package = {}
    for name in WORKER.CONTROL_FILES:
        content = (SCRIPTS / name).read_bytes()
        (control / name).write_bytes(content)
        target = release / "scripts" / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(content)
        package["scripts/" + name] = BACKUP.digest(target)
    write(release / "manifests/model.json", {"files": []})
    package["manifests/model.json"] = BACKUP.digest(release / "manifests/model.json")
    write(release / "PACKAGE_SHA256.json", package)
    dataset = tmp_path / "dapo.parquet"
    dataset.write_bytes(b"new authorized dataset")
    manifest = tmp_path / "dapo-manifest.json"
    write(manifest, {"contract": {"source_sha256": BACKUP.digest(dataset)}})
    native = tmp_path / "old-native-model"
    write(native / "study-assets.json", {"dataset_sha256": "old deepscaler dataset"})
    write(native / "tokenizer_config.json", {"chat_template": "native"})
    baseline = {
        "trainer_gpus": 4,
        "inference_gpus": 3,
        "inference_tensor_parallel": 1,
        "lag": 256,
        "historical_rollouts": str(tmp_path / "shared-queue"),
        "dataset_path": str(dataset),
        "data_manifest": str(manifest),
        "prepared_model_path": str(native),
        "model_path": str(tmp_path / "original-model"),
        "output_dir": str(tmp_path / "original-output"),
    }
    baseline_path = tmp_path / "baseline.json"
    write(baseline_path, baseline)
    spec = {
        "baseline_study": str(baseline_path),
        "release": str(release),
        "shared_prime": str(tmp_path / "shared-prime"),
        "workspace": str(workspace),
        "runtime": str(tmp_path / "runtime"),
        "run_name": "historical-worker",
    }
    write(control / "storage-spec.json", spec)
    return spec, control, baseline


def staged_receipt(spec):
    workspace, release = Path(spec["workspace"]), Path(spec["release"])
    baseline = json.loads(Path(spec["baseline_study"]).read_text())
    write(workspace / "study.json", LOCAL.relocate_study(baseline, workspace, spec["run_name"]))
    (workspace / "probe_allocated_gpus.py").write_bytes((SCRIPTS / "probe_allocated_gpus.py").read_bytes())
    return {
        "node": "deep-worker",
        "release": str(release),
        "python": str(Path(spec["runtime"]) / "prime-rl/.venv/bin/python"),
        "study_sha256": BACKUP.digest(workspace / "study.json"),
        "package_sha256": BACKUP.digest(release / "PACKAGE_SHA256.json"),
        "control_sha256": {"probe_allocated_gpus.py": BACKUP.digest(workspace / "probe_allocated_gpus.py")},
    }


def test_preparation_accepts_authorized_dataset_with_old_native_tokenizer_receipt(prepared):
    spec, control, baseline = prepared
    assert WORKER.validate_preparation(spec, control) == baseline


def test_preparation_rejects_changed_dataset_or_frozen_control(prepared):
    spec, control, baseline = prepared
    Path(baseline["dataset_path"]).write_bytes(b"unreviewed data")
    with pytest.raises(ValueError, match="dataset and prepared data manifest"):
        WORKER.validate_preparation(spec, control)
    (control / "node_local_run.py").write_text("modified")
    with pytest.raises(ValueError, match="frozen release"):
        WORKER.validate_preparation(spec, control)


@pytest.mark.parametrize("name", ["..", ".", "nested/worker", ""])
def test_preparation_rejects_output_path_escape(prepared, name):
    spec, control, _ = prepared
    with pytest.raises(ValueError, match="one path component"):
        WORKER.validate_preparation({**spec, "run_name": name}, control)


def test_worker_allocation_uses_only_inference_count_and_rejects_restarts():
    environment = allocation(3)
    assert WORKER.validate_allocation(environment, 3) == ["0", "1", "2"]
    for changed in ({"CUDA_VISIBLE_DEVICES": "0,1,1"}, {"SLURM_RESTART_COUNT": "1"}, {"SLURM_JOB_ID": ""}):
        with pytest.raises(ValueError, match="fresh Slurm allocation"):
            WORKER.validate_allocation({**environment, **changed}, 3)
    with pytest.raises(ValueError):
        WORKER.validate_allocation(environment, 7)


def test_worker_accepts_40gb_pool_without_collective():
    environment = allocation(3)
    probes = probes_for(environment)
    assert WORKER.validate_probes(probes, ["0", "1", "2"], environment) == ["0", "1", "2"]


@pytest.mark.parametrize(
    "change",
    [
        {"uuid": "GPU-1"},
        {"initial_free_bytes": 30_000_000_000},
        {"healthy": False},
        {"bf16_backward": False},
        {"name": "NVIDIA V100"},
        {"bytes": 20_000_000_000},
    ],
)
def test_worker_rejects_unhealthy_or_wrong_hardware(change):
    environment = allocation(3)
    probes = probes_for(environment)
    probes["results"][0].update(change)
    with pytest.raises(ValueError, match="healthy, unoccupied A100"):
        WORKER.validate_probes(probes, ["0", "1", "2"], environment)


def test_worker_environment_drops_cross_node_configuration(monkeypatch, tmp_path):
    original = {
        **allocation(3),
        "NCCL_NET": "Socket",
        "NCCL_IB_DISABLE": "1",
        "MASTER_ADDR": "other-node",
        "WORLD_SIZE": "12",
        "DEEPSEEK_STUDY_REMOTE_INFERENCE": "1",
        "TORCHELASTIC_RUN_ID": "old-run",
        "OMP_NUM_THREADS": "1",
    }
    monkeypatch.setattr(WORKER, "local_environment", lambda *_: original.copy())
    result = WORKER.worker_environment(tmp_path, tmp_path, tmp_path)
    assert result == {**allocation(3), "OMP_NUM_THREADS": "1"}
    assert original["NCCL_NET"] == "Socket"


def test_worker_staging_tamper_and_existing_output_fail(prepared, monkeypatch):
    spec, _, _ = prepared
    monkeypatch.setenv("SLURMD_NODENAME", "deep-worker")
    receipt = staged_receipt(spec)
    assert WORKER.verify_staged(spec, receipt) == Path(spec["release"])
    output = Path(json.loads((Path(spec["workspace"]) / "study.json").read_text())["output_dir"])
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="existing historical worker output"):
        WORKER.verify_staged(spec, receipt)
    output.rmdir()
    (Path(spec["release"]) / "scripts/local_backup.py").write_text("tampered")
    with pytest.raises(ValueError, match="source changed"):
        WORKER.verify_staged(spec, receipt)


def test_launch_stages_then_probes_then_execs_only_historical_worker(prepared, monkeypatch):
    spec, control, _ = prepared
    environment = allocation(3)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    calls = []

    def stage(received, received_control):
        assert received == spec and received_control == control
        calls.append("stage")
        return staged_receipt(spec)

    def command(args, **kwargs):
        calls.append("probe")
        assert Path(args[1]).name == "probe_allocated_gpus.py"
        assert "torch.distributed.run" not in args
        write(Path(args[args.index("--output") + 1]), probes_for(environment))

    def execute(python, args, env):
        calls.append("exec")
        assert args[1:4] == ["-m", "deepseek_study.runtime.historical_worker", "launch"]
        assert args[-2:] == ["--learner-job", "5678"]
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2"
        assert Path(python).is_relative_to(Path(spec["runtime"]))

    monkeypatch.setattr(WORKER, "stage", stage)
    monkeypatch.setattr(WORKER, "command", command)
    monkeypatch.setattr(WORKER, "worker_environment", lambda *_: environment.copy())
    monkeypatch.setattr(WORKER.os, "chdir", lambda _: None)
    monkeypatch.setattr(WORKER.os, "execve", execute)
    WORKER.launch(control, 5678)
    assert calls == ["stage", "probe", "exec"]
    receipt = json.loads((control / "historical-worker-launch-1234.json").read_text())
    assert receipt["trainer_devices"] == []
    assert receipt["cross_node_nccl"] is False
    assert receipt["inference_devices"] == ["0", "1", "2"]


def seal_fixture(prepared, monkeypatch):
    spec, control, baseline = prepared
    workspace = Path(spec["workspace"])
    release = workspace / "release"
    import shutil

    shutil.copytree(Path(spec["release"]), release)
    assets = workspace / "assets"
    assets.mkdir()
    shutil.copy2(baseline["dataset_path"], assets / "train.parquet")
    shutil.copy2(baseline["data_manifest"], assets / "train-manifest.json")
    native = assets / "native-model"
    native.mkdir()
    shutil.copy2(Path(baseline["prepared_model_path"]) / "tokenizer_config.json", native / "tokenizer_config.json")
    write(native / "study-assets.json", {"dataset_sha256": BACKUP.digest(baseline["dataset_path"])})
    write(workspace / "study.json", LOCAL.relocate_study(baseline, workspace, spec["run_name"]))
    monkeypatch.setattr(LOCAL, "secure_local", Path)
    monkeypatch.setattr(LOCAL, "configure_local_git", lambda *_: None)
    monkeypatch.setattr(LOCAL, "command", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(LOCAL, "local_environment", lambda *_: {})
    monkeypatch.setenv("SLURMD_NODENAME", "deep-worker")
    return spec, control, baseline


def test_seal_authorized_new_dataset_keeps_tokenizer_parity(prepared, monkeypatch):
    spec, control, baseline = seal_fixture(prepared, monkeypatch)
    receipt = LOCAL.seal_staging(spec, control)
    assert receipt["dataset_sha256"] == BACKUP.digest(baseline["dataset_path"])
    assert (
        receipt["dataset_sha256"]
        != json.loads((Path(baseline["prepared_model_path"]) / "study-assets.json").read_text())["dataset_sha256"]
    )


@pytest.mark.parametrize("changed", ["dataset", "receipt", "tokenizer"])
def test_seal_rejects_changed_staged_assets(prepared, monkeypatch, changed):
    spec, control, _ = seal_fixture(prepared, monkeypatch)
    assets = Path(spec["workspace"]) / "assets"
    path = {
        "dataset": assets / "train.parquet",
        "receipt": assets / "native-model/study-assets.json",
        "tokenizer": assets / "native-model/tokenizer_config.json",
    }[changed]
    write(path, {"dataset_sha256": "wrong"})
    with pytest.raises(ValueError, match="dataset|tokenizer"):
        LOCAL.seal_staging(spec, control)
    assert not (Path(spec["workspace"]) / "ready.json").exists()


def test_local_learner_uses_all_nine_gpus_and_explicit_40gb_floor():
    environment = allocation(9)
    probes = probes_for(environment)
    values = {"trainer_gpus": 4, "inference_gpus": 5}
    selected = LOCAL.select_local_devices({"minimum_gpu_bytes": 39_000_000_000}, values, probes, environment)
    assert selected == [str(i) for i in range(9)]
    with pytest.raises(ValueError, match="configured memory minimum"):
        LOCAL.select_local_devices({}, values, probes, environment)
    with pytest.raises(ValueError, match="allocation size"):
        LOCAL.select_local_devices({"allocated_gpus": 8}, values, probes, environment)


def test_local_learner_legacy_80gb_default_and_exact_allocation():
    environment = allocation(8)
    probes = probes_for(environment, 80_000_000_000)
    values = {"trainer_gpus": 4, "inference_gpus": 4}
    assert len(LOCAL.select_local_devices({}, values, probes, environment)) == 8
    probes["healthy_devices"].pop()
    with pytest.raises(ValueError, match="prepared topology"):
        LOCAL.select_local_devices({}, values, probes, environment)


def test_router_install_is_opt_in_and_checks_frozen_helper(prepared):
    spec, control, _ = prepared
    runtime, release = Path(spec["runtime"]), Path(spec["release"])
    LOCAL.install_stage_router(spec, control, runtime, release)
    assert not runtime.exists()
    with pytest.raises(ValueError, match="explicit boolean"):
        LOCAL.install_stage_router({**spec, "install_router": "yes"}, control, runtime, release)
    with pytest.raises(ValueError, match="frozen release"):
        LOCAL.install_stage_router({**spec, "install_router": True}, control, runtime, release)


def test_router_install_checks_wheel_before_writing_runtime(prepared):
    import zipfile

    spec, control, _ = prepared
    release, runtime = Path(spec["release"]), Path(spec["runtime"])
    wheel = control / "vllm_router-0.2.0-cp38-abi3-manylinux_2_28_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("vllm_router/__init__.py", "__version__ = '0.2.0'\n")
    helper = release / "scripts/multinode_run.py"
    helper.write_text(
        (SCRIPTS / "multinode_run.py")
        .read_text()
        .replace("bac193bedf10f9a0265fe4fdaae0f0418574cd1f15c45f27da1b4a2bae8c10b8", BACKUP.digest(wheel))
    )
    package = json.loads((release / "PACKAGE_SHA256.json").read_text())
    package["scripts/multinode_run.py"] = BACKUP.digest(helper)
    write(release / "PACKAGE_SHA256.json", package)
    enabled = {**spec, "install_router": True}
    LOCAL.install_stage_router(enabled, control, runtime, release)
    receipt = json.loads((runtime / "router-install.json").read_text())
    assert receipt == {"sha256": BACKUP.digest(wheel), "version": "0.2.0"}
    installed = runtime / "prime-rl/.venv/lib/python3.12/site-packages/vllm_router/__init__.py"
    assert installed.read_text() == "__version__ = '0.2.0'\n"
    wheel.write_bytes(b"changed wheel")
    with pytest.raises(ValueError, match="upstream lockfile"):
        LOCAL.install_stage_router(enabled, control, runtime, release)
    assert installed.read_text() == "__version__ = '0.2.0'\n"


def test_stage_installs_router_after_runtime_copy_and_before_seal(prepared, monkeypatch):
    import shutil
    import types

    spec, control, _ = prepared
    spec = {**spec, "install_router": True}
    calls = []
    monkeypatch.setattr(LOCAL, "secure_local", Path)
    monkeypatch.setattr(LOCAL.shutil, "disk_usage", lambda _: types.SimpleNamespace(free=200 * 1024**3))
    monkeypatch.setattr(LOCAL, "copy_tree", lambda source, target, _: shutil.copytree(source, target))

    def relocate(*_):
        calls.append("runtime")
        return Path(spec["runtime"]) / "python"

    def install(*args):
        assert args[0]["install_router"] is True
        assert calls == ["runtime"]
        calls.append("router")

    def seal(*_):
        calls.append("seal")
        return {"sealed": True}

    monkeypatch.setattr(LOCAL, "relocate_runtime", relocate)
    monkeypatch.setattr(LOCAL, "install_stage_router", install)
    monkeypatch.setattr(LOCAL, "local_environment", lambda *_: {})
    monkeypatch.setattr(LOCAL, "command", lambda *_args, **_kwargs: calls.append("prepare"))
    monkeypatch.setattr(LOCAL, "seal_staging", seal)
    assert LOCAL.stage(spec, control) == {"sealed": True}
    assert calls == ["runtime", "router", "prepare", "seal"]
