import json
import sys
from types import SimpleNamespace

import pytest
import torch

from deepseek_study.config import StudyConfig
from deepseek_study.dataset import assets
from deepseek_study.runtime import historical_worker, identity, launcher, processes
from deepseek_study.runtime.build import build, resolve
from prime_rl.entrypoints import rl


@pytest.mark.parametrize("inference_gpus", [1, 3, 5])
def test_historical_pool_routes_generations_and_admin_updates_to_distinct_endpoints(study, tmp_path, inference_gpus):
    study = StudyConfig.model_validate(
        study.model_dump()
        | {
            "historical_rollouts": tmp_path / "history",
            "inference_gpus": inference_gpus,
            "inference_tensor_parallel": 1,
        }
    )
    configuration = build(study, tmp_path / "configs")
    expected_admin = [f"http://127.0.0.1:{study.inference_port + 100 + index}/v1" for index in range(inference_gpus)]
    assert configuration.orchestrator.model.client.base_url == f"http://127.0.0.1:{study.inference_port}/v1"
    assert configuration.orchestrator.model.client.admin_base_url == expected_admin
    assert configuration.inference.server.port == study.inference_port
    assert configuration.inference.vllm.data_parallel_size == inference_gpus
    assert configuration.inference.vllm.tensor_parallel_size == 1
    written = json.loads((tmp_path / "configs/orchestrator.json").read_text())
    assert written["model"]["client"]["admin_base_url"] == expected_admin
    assert configuration.inference.weight_broadcast.type == "filesystem"


def test_historical_pool_rejects_tensor_parallel_configuration(study, tmp_path):
    study = study.model_copy(update={"historical_rollouts": tmp_path / "history"})
    with pytest.raises(ValueError, match="tensor parallel size 1"):
        resolve(study)


def test_legacy_inference_keeps_native_server_and_admin_defaults(study, tmp_path):
    configuration = resolve(study)
    assert configuration.inference.vllm.tensor_parallel_size == study.inference_tensor_parallel
    assert configuration.orchestrator.model.client.admin_base_url is None
    assert launcher.inference_command(study, tmp_path / "inference.json") == [
        sys.executable,
        "-m",
        "prime_rl.entrypoints.inference",
        "@",
        str(tmp_path / "inference.json"),
    ]


@pytest.mark.parametrize("historical_worker_launch", [False, True])
def test_both_historical_launchers_start_pool_with_owned_gpu_allocation(
    study, tmp_path, monkeypatch, historical_worker_launch
):
    study = study.model_copy(update={"historical_rollouts": tmp_path / "history", "inference_tensor_parallel": 1})
    study.historical_rollouts.mkdir()
    (study.historical_rollouts / "session.json").write_text("{}")
    allocated = ["GPU-physical-A", "GPU-physical-B"]
    if not historical_worker_launch:
        allocated.extend(["GPU-physical-C", "GPU-physical-D"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(allocated))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("DEEPSEEK_STUDY_REMOTE_INFERENCE", raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(allocated))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "mock")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(total_memory=1))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 0))
    monkeypatch.setattr(launcher, "verify_upstream", lambda root: None)
    monkeypatch.setattr(launcher, "configure_nccl_transport", lambda: None)
    monkeypatch.setattr(launcher, "validate_prepared", lambda study: {})
    monkeypatch.setattr(assets, "validate_prepared", lambda study: {})
    source = {"sha256": "source", "runtime": {"vllm": "test"}}
    monkeypatch.setattr(identity, "capture", lambda root, study: source)
    monkeypatch.setattr(launcher, "capture", lambda root, study: source)
    monkeypatch.setattr(identity, "snapshot", lambda *args: None)
    monkeypatch.setattr(launcher, "snapshot", lambda *args: None)
    monkeypatch.setattr(launcher, "env_servers", lambda config: ())
    monkeypatch.setattr(rl, "env_servers", lambda config: ())
    monkeypatch.setattr(historical_worker, "validate_session", lambda *args: None)
    monkeypatch.setattr(launcher, "drain_process", lambda *args: (True, []))
    cleaned = []
    monkeypatch.setattr(processes, "stop_process_groups", lambda owned: cleaned.extend(owned) or [])
    monkeypatch.setattr(launcher, "stop_process_groups", lambda owned: cleaned.extend(owned) or [])
    started = []

    class Process:
        def __init__(self, args, **kwargs):
            self.args = args
            self.options = kwargs
            self.pid = 1000 + len(started)
            self.returncode = 0 if "controller" in args or "consume" in args or "torch.distributed.run" in args else None
            started.append(self)
            if "controller" in args:
                (study.output_dir / "study-complete.json").write_text("{}")

        def poll(self):
            return self.returncode

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    if historical_worker_launch:
        historical_worker.launch(study, tmp_path, 122)
    else:
        launcher.launch(study, tmp_path)
    pool = next(process for process in started if "deepseek_study.runtime.inference_pool" in process.args)
    assert pool.args == [
        sys.executable,
        "-m",
        "deepseek_study.runtime.inference_pool",
        str(study.output_dir / "configs/inference.json"),
    ]
    assert pool.options["env"]["CUDA_VISIBLE_DEVICES"] == ",".join(allocated[: study.inference_gpus])
    assert pool.options["start_new_session"]
    assert "inference" in cleaned
    assert not any("prime_rl.entrypoints.inference" in process.args for process in started)
