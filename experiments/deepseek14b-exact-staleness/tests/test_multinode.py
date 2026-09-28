import importlib.util
import json
import sys
from pathlib import Path

import pytest

from deepseek_study.config import StudyConfig
from deepseek_study.runtime.build import build, resolve
from deepseek_study.runtime.deployment import RemoteInference


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("multinode_run", SCRIPTS / "multinode_run.py")
multinode = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multinode)


def remote():
    return RemoteInference(**multinode.layout(["node-a", "node-b"], 8000), hardware=[])


def distributed(study):
    return StudyConfig.model_validate(
        study.model_dump() | {"trainer_gpus": 4, "inference_gpus": 8, "inference_tensor_parallel": 1}
    )


def test_remote_configs_wire_all_workers_and_keep_science(study, tmp_path):
    study = distributed(study)
    local = resolve(study)
    result = build(study, tmp_path / "configs", remote=remote())
    trainer = json.loads((tmp_path / "configs/trainer.json").read_text())
    controller = json.loads((tmp_path / "configs/orchestrator.json").read_text())
    assert trainer["weight_broadcast"]["host"] == "node-a"
    assert trainer["weight_broadcast"]["inference_world_size"] == 8
    assert controller["weight_broadcast"]["inference_world_size"] == 8
    assert controller["model"]["client"]["admin_base_url"] == remote().worker_urls
    assert controller["model"]["client"]["base_url"] == "http://node-a:8000/v1"
    assert result.trainer.loss == local.trainer.loss
    assert result.trainer.optim == local.trainer.optim
    assert result.orchestrator.train == local.orchestrator.train
    assert result.orchestrator.num_train_workers == 4
    assert result.trainer.rollout_transport == local.trainer.rollout_transport


def test_duplicate_or_missing_endpoints_fail(study):
    values = remote().model_dump()
    values["worker_urls"][-1] = values["worker_urls"][0]
    with pytest.raises(ValueError, match="unique"):
        RemoteInference(**values)
    with pytest.raises(ValueError, match="one registered endpoint"):
        remote().validate_study(study)


def test_all_twelve_devices_have_exactly_one_role():
    result = multinode.layout(["node-a", "node-b"], 8000)
    assert len(result["worker_urls"]) == len(set(result["worker_urls"])) == 8
    assert sum("node-a" in url for url in result["worker_urls"]) == 2
    assert sum("node-b" in url for url in result["worker_urls"]) == 6
    with pytest.raises(ValueError):
        multinode.layout(["same", "same"], 8000)


def test_failed_or_small_gpu_is_never_silently_dropped():
    devices = [str(i) for i in range(6)]
    receipt = {
        "allocated_devices": devices,
        "healthy_devices": devices,
        "results": [{"name": "A100", "bytes": 40_000_000_000} for _ in devices],
    }
    assert multinode.validate_health(receipt) == devices
    receipt["healthy_devices"] = devices[:-1]
    with pytest.raises(ValueError, match="Every allocated GPU"):
        multinode.validate_health(receipt)


def test_inference_engines_use_distinct_reproducible_rng_streams():
    seeds = [multinode.worker_seed(42, node, rank) for node, count in ((0, 2), (1, 6)) for rank in range(count)]
    assert seeds == list(range(42, 50))
    with pytest.raises(ValueError):
        multinode.worker_seed(42, 0, 2)


@pytest.mark.parametrize("label", ["eth0", "enx1234"])
def test_network_pins_the_routed_ipv4_address_label(monkeypatch, label):
    monkeypatch.setattr(multinode.socket, "gethostbyname", lambda _: "10.0.0.2")

    def output(command, **kwargs):
        if "route" in command:
            return '[{"dev":"eth0","prefsrc":"10.0.0.1"}]'
        return json.dumps([{"ifname": "eth0", "addr_info": [{"family": "inet", "local": "10.0.0.1", "label": label}]}])

    monkeypatch.setattr(multinode.subprocess, "check_output", output)
    assert multinode.network_environment(["a", "b"], 0) == {
        "NCCL_NET": "Socket",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SOCKET_FAMILY": "AF_INET",
        "NCCL_SOCKET_IFNAME": "=" + label,
        "GLOO_SOCKET_IFNAME": label,
        "VLLM_HOST_IP": "10.0.0.1",
    }
    monkeypatch.setattr(multinode.subprocess, "check_output", lambda *args, **kwargs: "[]")
    with pytest.raises(ValueError, match="IPv4 route"):
        multinode.network_environment(["a", "b"], 0)


def test_backup_command_runs_the_actual_cli_before_training(tmp_path):
    import subprocess

    workspace = tmp_path / "local"
    workspace.mkdir()
    (workspace / "backup.stop").touch()
    command = multinode.backup_command(
        sys.executable,
        SCRIPTS,
        workspace / "run",
        {"backup": str(tmp_path / "shared"), "metrics": str(tmp_path / "metrics")},
        workspace,
        tmp_path / "runtime",
    )
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads((workspace / "backup-status.json").read_text()) == {"status": "waiting_for_run", "final": True}


def test_workers_must_all_serve_the_expected_model(monkeypatch):
    urls = remote().worker_urls
    states = {url + "/models": {"data": [{"id": "expected"}]} for url in urls}
    monkeypatch.setattr(multinode, "fetch_json", lambda url: states[url])
    assert multinode.workers_ready(urls, "expected")
    states[urls[-1] + "/models"] = {"data": [{"id": "wrong-model"}]}
    assert not multinode.workers_ready(urls, "expected")
    del states[urls[-1] + "/models"]
    assert not multinode.workers_ready(urls, "expected")


def test_router_rejects_early_unknown_or_missing_workers(monkeypatch):
    deployment = remote()
    rows = [
        {"url": url.removesuffix("/v1"), "is_healthy": True, "model_id": "expected"} for url in deployment.worker_urls
    ]
    monkeypatch.setattr(multinode, "fetch_json", lambda url: {"workers": rows})
    assert multinode.router_ready(deployment.router_url, deployment.worker_urls, "expected")
    rows[-1]["model_id"] = "unknown"
    assert not multinode.router_ready(deployment.router_url, deployment.worker_urls, "expected")
    rows[-1]["model_id"] = "expected"
    rows[-1]["is_healthy"] = False
    assert not multinode.router_ready(deployment.router_url, deployment.worker_urls, "expected")
    rows.pop()
    assert not multinode.router_ready(deployment.router_url, deployment.worker_urls, "expected")


def test_first_checkpoint_dispatch_preserves_periodic_schedule(study):
    study = distributed(study)
    study.checkpoint_first_step = True
    study.checkpoint_interval = 5
    config = resolve(study, remote=remote())
    assert config.trainer.ckpt.interval == 1
    assert study.checkpoint_due(1)
    assert not study.checkpoint_due(2)
    assert study.checkpoint_due(5)
