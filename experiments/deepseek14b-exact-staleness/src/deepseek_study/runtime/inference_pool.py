import argparse
import fcntl
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

from prime_rl.configs.inference import InferenceConfig

from deepseek_study.runtime.checkpoints import atomic_write
from deepseek_study.runtime.processes import stop_process_groups

ROUTER_VERSION = "0.2.0"
CACHE_KEYS = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "VLLM_RPC_BASE_PATH")
DISTRIBUTED_KEYS = {
    "RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NODE_RANK",
    "RANK_ID",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NTASKS",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_NODEID",
    "VLLM_PORT",
}
DISTRIBUTED_PREFIXES = ("TORCHELASTIC_", "OMPI_", "PMI_", "PMIX_", "VLLM_DP_")


def clean_environment(environment):
    return {
        key: value
        for key, value in environment.items()
        if key not in DISTRIBUTED_KEYS and not key.startswith(DISTRIBUTED_PREFIXES)
    }


def worker_urls(config):
    count = config.vllm.data_parallel_size
    if not 1 <= count <= 32 or not 1 <= config.server.port <= 65535 - 200 - count:
        raise ValueError("Inference pool size or reserved port range is invalid")
    return [f"http://127.0.0.1:{config.server.port + 100 + index}/v1" for index in range(count)]


def prepare_workers(config, environment):
    urls = worker_urls(config)
    devices = [device.strip() for device in environment.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    if len(devices) != len(urls) or not all(devices) or len(set(devices)) != len(devices) or "-1" in devices:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain exactly one distinct allocated GPU per inference replica")
    if config.vllm.tensor_parallel_size != 1 or config.weight_broadcast.type != "filesystem":
        raise ValueError("Independent inference replicas require TP1 and filesystem weight loading")
    if config.slurm is not None or config.dry_run or config.deployment.type != "single_node":
        raise ValueError("Inference pool requires an active local single-node configuration")
    merged = clean_environment({**environment, **config.env_vars})
    workers = []
    for index, device in enumerate(devices):
        overrides = {"CUDA_VISIBLE_DEVICES": device, "VLLM_HOST_IP": "127.0.0.1"}
        for key in CACHE_KEYS:
            root = Path(merged.get(key) or config.output_dir / "inference-pool-cache" / key.lower())
            overrides[key] = str(root / f"worker-{index}")
        payload = config.model_dump(mode="json")
        payload["server"].update(host="127.0.0.1", port=config.server.port + 100 + index)
        payload["router"] = None
        payload["deployment"] = {"type": "single_node", "gpus_per_node": 1}
        payload["output_dir"] = str(config.output_dir / "inference-pool" / f"worker-{index}")
        payload["env_vars"] = {
            **clean_environment(config.env_vars),
            **{key: value for key, value in overrides.items() if key != "CUDA_VISIBLE_DEVICES"},
        }
        payload["env_vars"].pop("CUDA_VISIBLE_DEVICES", None)
        vllm = payload["vllm"]
        for key in tuple(vllm):
            if key.startswith("data_parallel_") and key not in {
                "data_parallel_size",
                "data_parallel_size_local",
                "data_parallel_rpc_port",
            }:
                del vllm[key]
        if vllm.pop("headless", False) or vllm.get("pipeline_parallel_size", 1) != 1:
            raise ValueError("Independent inference replicas cannot use headless or pipeline parallel inference")
        vllm.update(
            tensor_parallel_size=1,
            data_parallel_size=1,
            data_parallel_size_local=1,
            api_server_count=1,
            data_parallel_rpc_port=config.server.port + 200 + index,
            seed=config.vllm.seed + index,
        )
        replica = InferenceConfig.model_validate(payload)
        workers.append((replica, {**merged, **overrides}))
    return workers


def router_command(config):
    return [
        sys.executable,
        "-m",
        "vllm_router.launch_router",
        "--host",
        "127.0.0.1",
        "--port",
        str(config.server.port),
        "--policy",
        "round_robin",
        "--worker-urls",
        *[url.removesuffix("/v1") for url in worker_urls(config)],
        "--worker-startup-timeout-secs",
        "1800",
        "--retry-max-retries",
        "1",
        "--disable-retries",
        "--request-timeout-secs",
        "86400",
        "--request-id-headers",
        "x-session-id",
    ]


def check_ports_available(config):
    ports = [
        config.server.port,
        *[config.server.port + 100 + index for index in range(config.vllm.data_parallel_size)],
    ]
    with ExitStack() as stack:
        for port in ports:
            listener = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                listener.bind(("127.0.0.1", port))
                listener.listen(1)
            except OSError as error:
                raise RuntimeError(f"Inference pool port {port} is already occupied or unavailable") from error


def fetch_json(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=3) as response:
        return json.load(response)


def workers_ready(urls, model):
    def ready(url):
        try:
            return model in {item["id"] for item in fetch_json(url + "/models")["data"]}
        except (OSError, ValueError, KeyError, TypeError):
            return False

    with ThreadPoolExecutor(max_workers=len(urls)) as executor:
        return all(executor.map(ready, urls))


def router_ready(url, workers, model):
    try:
        rows = fetch_json(url.removesuffix("/v1") + "/workers")["workers"]
        return (
            len(rows) == len(workers)
            and {row["url"] for row in rows} == {worker.removesuffix("/v1") for worker in workers}
            and all(row["is_healthy"] and row["model_id"] == model for row in rows)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def check_children(processes):
    for name, process in processes.items():
        status = process.poll()
        if status is not None:
            raise RuntimeError(f"Inference pool child {name} exited with status {status}; see its pool log")


def wait_ready(predicate, processes, timeout):
    deadline = time.monotonic() + timeout
    while True:
        check_children(processes)
        ready = predicate()
        check_children(processes)
        if ready:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Inference pool readiness exceeded {timeout} seconds")
        time.sleep(min(1, max(0, deadline - time.monotonic())))


def supervise(config, environment=None, worker_timeout=1800, router_timeout=180):
    environment = dict(os.environ if environment is None else environment)
    workers = prepare_workers(config, environment)
    version = importlib.metadata.version("vllm-router")
    if version != ROUTER_VERSION:
        raise ValueError(f"Inference pool requires vllm-router {ROUTER_VERSION}; found {version}")
    configs = config.output_dir / "configs" / "inference-pool"
    logs = config.output_dir / "logs" / "inference-pool"
    configs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    receipt = config.output_dir / "inference-pool-ready.json"
    processes, streams, handlers = {}, [], {}
    spawning, pending_signal = False, None
    lock = (configs / "pool.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise RuntimeError("Another inference pool owns this output directory") from None

    def interrupt(signum, _frame):
        nonlocal pending_signal
        if spawning:
            pending_signal = signum
            return
        raise SystemExit(128 + signum)

    def start(name, argv, child_environment):
        nonlocal spawning
        stream = (logs / f"{name}.log").open("a", buffering=1)
        streams.append(stream)
        spawning = True
        try:
            processes[name] = subprocess.Popen(
                argv,
                env=child_environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            spawning = False
        if pending_signal is not None:
            raise SystemExit(128 + pending_signal)

    try:
        receipt.unlink(missing_ok=True)
        check_ports_available(config)
        for signum in (signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, interrupt)
        for index, (replica, child_environment) in enumerate(workers):
            for key in CACHE_KEYS:
                Path(child_environment[key]).mkdir(parents=True, exist_ok=True)
            path = configs / f"worker-{index}.json"
            atomic_write(path, (replica.model_dump_json(indent=2) + "\n").encode())
            start(
                f"worker-{index}",
                [sys.executable, "-m", "prime_rl.entrypoints.inference", "@", str(path)],
                child_environment,
            )
        urls = worker_urls(config)
        wait_ready(lambda: workers_ready(urls, config.vllm.model), processes, worker_timeout)
        router_environment = clean_environment({**environment, **config.env_vars, "CUDA_VISIBLE_DEVICES": ""})
        start("router", router_command(config), router_environment)
        wait_ready(
            lambda: router_ready(f"http://127.0.0.1:{config.server.port}/v1", urls, config.vllm.model),
            processes,
            router_timeout,
        )
        atomic_write(
            receipt,
            (
                json.dumps(
                    {
                        "router_version": version,
                        "router_policy": "round_robin",
                        "max_request_attempts": 1,
                        "router_url": f"http://127.0.0.1:{config.server.port}/v1",
                        "workers": urls,
                        "devices": [worker_environment["CUDA_VISIBLE_DEVICES"] for _, worker_environment in workers],
                        "seeds": [replica.vllm.seed for replica, _ in workers],
                        "model": config.vllm.model,
                    },
                    indent=2,
                )
                + "\n"
            ).encode(),
        )
        print(f"Independent inference pool ready: {len(workers)} replicas on port {config.server.port}", flush=True)
        while True:
            check_children(processes)
            time.sleep(1)
    finally:
        for signum in handlers:
            signal.signal(signum, signal.SIG_IGN)
        errors = stop_process_groups(processes, graceful_timeout=10, kill_timeout=3)
        receipt.unlink(missing_ok=True)
        for stream in streams:
            stream.close()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        lock.close()
        for error in errors:
            print(error, file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    supervise(InferenceConfig.model_validate_json(args.config.read_text()))


if __name__ == "__main__":
    main()
