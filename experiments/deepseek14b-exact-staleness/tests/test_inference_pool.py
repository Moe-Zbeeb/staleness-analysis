import json
import signal
import socket
import sys
from types import SimpleNamespace

import pytest
from prime_rl.configs.inference import InferenceConfig

from deepseek_study.runtime import inference_pool as pool


@pytest.fixture
def config(tmp_path):
    return InferenceConfig(
        server={"host": "0.0.0.0", "port": 18000},
        output_dir=tmp_path,
        vllm={
            "model": "/local/model",
            "data_parallel_size": 5,
            "data_parallel_size_local": 5,
            "tensor_parallel_size": 1,
            "seed": 42,
            "max_model_len": 8192,
            "max_num_seqs": 64,
            "enable_prefix_caching": False,
            "dtype": "bfloat16",
            "generation_config": "vllm",
            "logprobs_mode": "raw_logprobs",
            "tool_call_parser": None,
            "reasoning_parser": None,
        },
    )


def test_replicas_preserve_sampling_and_weights_while_isolating_physical_devices(config):
    before = config.model_dump(mode="json")
    environment = {"CUDA_VISIBLE_DEVICES": "GPU-a,3,4,6,8", "RANK": "2", "TOKEN_SECRET": "hidden"}
    workers = pool.prepare_workers(config, environment)
    assert config.model_dump(mode="json") == before
    assert environment["RANK"] == "2"
    assert [env["CUDA_VISIBLE_DEVICES"] for _, env in workers] == ["GPU-a", "3", "4", "6", "8"]
    for index, (replica, env) in enumerate(workers):
        assert replica.server.host == "127.0.0.1"
        assert replica.server.port == 18100 + index
        assert replica.router is None
        assert replica.deployment.gpus_per_node == 1
        assert replica.vllm.tensor_parallel_size == replica.vllm.data_parallel_size == 1
        assert replica.vllm.data_parallel_size_local == replica.vllm.api_server_count == 1
        assert replica.vllm.data_parallel_rpc_port == 18200 + index
        assert replica.vllm.seed == 42 + index
        assert replica.vllm.max_model_len == 8192
        assert replica.vllm.model_extra["max_num_seqs"] == 64
        assert replica.vllm.model_extra["generation_config"] == "vllm"
        assert replica.vllm.model_extra["logprobs_mode"] == "raw_logprobs"
        assert replica.vllm.dtype == "bfloat16"
        assert replica.vllm.reasoning_parser is replica.vllm.tool_call_parser is None
        assert replica.weight_broadcast.type == "filesystem"
        assert replica.vllm.enable_prefix_caching is False
        assert "RANK" not in env
        assert env["TOKEN_SECRET"] == "hidden"
        assert "TOKEN_SECRET" not in replica.env_vars
        assert "CUDA_VISIBLE_DEVICES" not in replica.env_vars


def test_distributed_environment_and_compilation_caches_are_isolated(config, tmp_path):
    config.env_vars = {"LOCAL_RANK": "3", "TRITON_CACHE_DIR": str(tmp_path / "triton")}
    environment = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4",
        "RANK": "9",
        "WORLD_SIZE": "12",
        "SLURM_JOB_ID": "123",
        "TORCHELASTIC_RUN_ID": "bad",
        "PMI_RANK": "2",
        "OMPI_COMM_WORLD_SIZE": "12",
        "SLURM_PROCID": "3",
        "VLLM_DP_RANK": "2",
        "VLLM_HOST_IP": "192.0.2.1",
        "VLLM_PORT": "5000",
        "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor"),
    }
    workers = pool.prepare_workers(config, environment)
    for index, (replica, env) in enumerate(workers):
        for key in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "TORCHELASTIC_RUN_ID",
            "PMI_RANK",
            "OMPI_COMM_WORLD_SIZE",
            "SLURM_PROCID",
            "VLLM_DP_RANK",
            "VLLM_PORT",
        ):
            assert key not in env
            assert key not in replica.env_vars
        assert env["SLURM_JOB_ID"] == "123"
        assert env["VLLM_HOST_IP"] == "127.0.0.1"
        assert env["TRITON_CACHE_DIR"] == str(tmp_path / "triton" / f"worker-{index}")
        assert env["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "inductor" / f"worker-{index}")
        for key in pool.CACHE_KEYS:
            assert env[key] == replica.env_vars[key]
    assert len({env["VLLM_RPC_BASE_PATH"] for _, env in workers}) == 5


@pytest.mark.parametrize("devices", ["", "0,1", "0,1,2,3,3", "0,1,2,3,", "0,1,2,3,-1"])
def test_wrong_gpu_allocation_fails_closed(config, devices):
    with pytest.raises(ValueError, match="distinct allocated GPU"):
        pool.prepare_workers(config, {"CUDA_VISIBLE_DEVICES": devices})


@pytest.mark.parametrize("change", ["tp", "transport", "port", "dry_run", "headless"])
def test_unsupported_topology_fails_before_spawning(config, change):
    if change == "tp":
        config.vllm.tensor_parallel_size = 2
    elif change == "transport":
        config.weight_broadcast.type = "nccl"
    elif change == "port":
        config.server.port = 65500
    elif change == "dry_run":
        config.dry_run = True
    else:
        config.vllm.model_extra["headless"] = True
    with pytest.raises(ValueError):
        pool.prepare_workers(config, {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4"})


def test_router_uses_pinned_single_attempt_and_bare_worker_urls(config):
    command = pool.router_command(config)
    assert command[:3] == [sys.executable, "-m", "vllm_router.launch_router"]
    assert command[command.index("--retry-max-retries") + 1] == "1"
    assert "--disable-retries" in command
    assert command[command.index("--policy") + 1] == "round_robin"
    assert "http://127.0.0.1:18100" in command
    assert not any(value.endswith("/v1") for value in command)
    assert command[-2:] == ["--request-id-headers", "x-session-id"]


def test_router_readiness_requires_every_expected_healthy_replica(config, monkeypatch):
    urls = pool.worker_urls(config)
    rows = [{"url": url.removesuffix("/v1"), "is_healthy": True, "model_id": config.vllm.model} for url in urls]
    monkeypatch.setattr(pool, "fetch_json", lambda url: {"workers": rows})
    assert pool.router_ready("http://127.0.0.1:18000/v1", urls, config.vllm.model)
    rows[0]["is_healthy"] = False
    assert not pool.router_ready("http://127.0.0.1:18000/v1", urls, config.vllm.model)
    rows[0]["is_healthy"] = True
    rows[0]["model_id"] = "wrong-model"
    assert not pool.router_ready("http://127.0.0.1:18000/v1", urls, config.vllm.model)


def test_worker_readiness_checks_model_on_each_replica(config, monkeypatch):
    urls = pool.worker_urls(config)
    monkeypatch.setattr(pool, "fetch_json", lambda url: {"data": [{"id": config.vllm.model}]})
    assert pool.workers_ready(urls, config.vllm.model)
    monkeypatch.setattr(pool, "fetch_json", lambda url: {"data": [{"id": "wrong"}]})
    assert not pool.workers_ready(urls, config.vllm.model)


def test_readiness_rejects_dead_children_even_with_a_healthy_endpoint():
    with pytest.raises(RuntimeError, match="worker-0 exited with status 0"):
        pool.wait_ready(lambda: True, {"worker-0": SimpleNamespace(poll=lambda: 0)}, 1)


def test_readiness_has_a_deadline():
    with pytest.raises(TimeoutError, match="readiness exceeded"):
        pool.wait_ready(lambda: False, {}, 0)


@pytest.mark.parametrize("failure", ["startup", "serving", "signal"])
def test_supervisor_stops_only_owned_groups_and_restores_signal_handlers(config, monkeypatch, failure):
    launched, stopped = [], []
    initial_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    monkeypatch.setattr(pool.importlib.metadata, "version", lambda name: "0.2.0")
    monkeypatch.setattr(pool, "check_ports_available", lambda config: None)

    def popen(argv, **kwargs):
        child = SimpleNamespace(poll=lambda: None, pid=1000 + len(launched))
        launched.append((argv, kwargs, child))
        return child

    monkeypatch.setattr(pool.subprocess, "Popen", popen)
    monkeypatch.setattr(pool, "stop_process_groups", lambda processes, **kwargs: stopped.append(dict(processes)) or [])

    def ready(*args):
        if failure == "startup":
            raise TimeoutError("startup failed")
        return True

    monkeypatch.setattr(pool, "workers_ready", ready)
    monkeypatch.setattr(pool, "router_ready", lambda *args: True)

    def sleep(seconds):
        receipt = json.loads((config.output_dir / "inference-pool-ready.json").read_text())
        assert receipt["seeds"] == [42, 43, 44, 45, 46]
        assert receipt["max_request_attempts"] == 1
        assert receipt["devices"] == ["0", "1", "2", "3", "4"]
        if failure == "signal":
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        raise RuntimeError("serving failed")

    monkeypatch.setattr(pool.time, "sleep", sleep)
    error = {"startup": TimeoutError, "serving": RuntimeError, "signal": SystemExit}[failure]
    with pytest.raises(error):
        pool.supervise(config, {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4", "RANK": "9"})
    assert len(launched) == (5 if failure == "startup" else 6)
    assert len(stopped[0]) == len(launched)
    for index, (argv, kwargs, child) in enumerate(launched[:5]):
        assert argv[:4] == [sys.executable, "-m", "prime_rl.entrypoints.inference", "@"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == str(index)
        assert "RANK" not in kwargs["env"]
        assert kwargs["start_new_session"] is True
        assert kwargs["stdout"].closed
        assert stopped[0][f"worker-{index}"] is child
    if failure != "startup":
        assert launched[-1][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert not (config.output_dir / "inference-pool-ready.json").exists()
    assert {signum: signal.getsignal(signum) for signum in initial_handlers} == initial_handlers


def test_router_version_mismatch_prevents_any_spawn(config, monkeypatch):
    monkeypatch.setattr(pool.importlib.metadata, "version", lambda name: "0.3.0")
    monkeypatch.setattr(pool.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not spawn"))
    with pytest.raises(ValueError, match="requires vllm-router 0.2.0"):
        pool.supervise(config, {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4"})


def test_signal_during_spawn_registers_child_before_cleanup(config, monkeypatch):
    child = SimpleNamespace(poll=lambda: None, pid=4321)
    stopped = []
    monkeypatch.setattr(pool.importlib.metadata, "version", lambda name: "0.2.0")
    monkeypatch.setattr(pool, "check_ports_available", lambda config: None)

    def popen(*args, **kwargs):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return child

    def stop(processes, **kwargs):
        stopped.append(dict(processes))
        assert kwargs == {"graceful_timeout": 10, "kill_timeout": 3}
        return []

    monkeypatch.setattr(pool.subprocess, "Popen", popen)
    monkeypatch.setattr(pool, "stop_process_groups", stop)
    with pytest.raises(SystemExit, match=str(128 + signal.SIGTERM)):
        pool.supervise(config, {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4"})
    assert stopped == [{"worker-0": child}]


def test_mutated_config_gpu_override_cannot_replace_allocated_device(config):
    config.env_vars["CUDA_VISIBLE_DEVICES"] = "unallocated-device"
    workers = pool.prepare_workers(config, {"CUDA_VISIBLE_DEVICES": "0,2,4,6,8"})
    for (replica, environment), device in zip(workers, ["0", "2", "4", "6", "8"], strict=True):
        assert environment["CUDA_VISIBLE_DEVICES"] == device
        assert "CUDA_VISIBLE_DEVICES" not in replica.env_vars
        reapplied = {**environment, **replica.env_vars}
        assert reapplied["CUDA_VISIBLE_DEVICES"] == device


def test_preexisting_server_port_is_not_accepted_as_new_pool_readiness(config):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        config.server.port = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match="already occupied or unavailable"):
            pool.check_ports_available(config)
