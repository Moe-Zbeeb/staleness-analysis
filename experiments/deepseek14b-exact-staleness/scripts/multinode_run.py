import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import zipfile
from pathlib import Path

from local_backup import atomic_json, digest
from node_local_run import command, local_environment, stage


ROUTER_SHA256 = "bac193bedf10f9a0265fe4fdaae0f0418574cd1f15c45f27da1b4a2bae8c10b8"


def layout(hosts, port):
    if len(hosts) != 2 or len(set(hosts)) != 2:
        raise ValueError("This layout requires two distinct six-GPU nodes")
    return {
        "trainer_host": hosts[0],
        "router_url": f"http://{hosts[0]}:{port}/v1",
        "worker_urls": [
            f"http://{host}:{port + 100 + rank}/v1"
            for host, count in zip(hosts, (2, 6), strict=True)
            for rank in range(count)
        ],
    }


def worker_seed(seed, node_rank, worker_rank):
    count = 2 if node_rank == 0 else 6
    if node_rank not in (0, 1) or not 0 <= worker_rank < count:
        raise ValueError("Invalid inference worker assignment")
    return seed + worker_rank + (2 if node_rank else 0)


def install_router(control, runtime):
    wheel = control / "vllm_router-0.2.0-cp38-abi3-manylinux_2_28_x86_64.whl"
    if digest(wheel) != ROUTER_SHA256:
        raise ValueError("Router wheel differs from the upstream lockfile")
    site = runtime / "prime-rl/.venv/lib/python3.12/site-packages"
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not (site / name).resolve().is_relative_to(site.resolve()):
                raise ValueError("Invalid wheel member")
        archive.extractall(site)
    atomic_json(runtime / "router-install.json", {"sha256": ROUTER_SHA256, "version": "0.2.0"})


def stop_process(process, seconds=120):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def validate_health(receipt):
    rows = receipt["results"]
    devices = receipt["allocated_devices"]
    if len(devices) != 6 or len(set(devices)) != 6 or len(rows) != 6:
        raise ValueError("Exactly six allocated GPUs are required per node")
    if set(receipt["healthy_devices"]) != set(devices):
        raise ValueError("Every allocated GPU must pass health checks; no silent fallback")
    if any("A100" not in row["name"] or row["bytes"] < 39_000_000_000 for row in rows):
        raise ValueError("All allocated GPUs must be A100 40GB or larger")
    return devices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    control = args.control.resolve()
    for name, expected in json.loads((control / "CONTROL_SHA256.json").read_text()).items():
        if digest(control / name) != expected:
            raise ValueError(f"Launch control changed: {name}")
    rank = int(os.environ["SLURM_PROCID"])
    hosts = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True
    ).split()
    if rank not in (0, 1) or len(hosts) != 2 or int(os.environ.get("SLURM_RESTART_COUNT", "0")):
        raise ValueError("Unexpected allocation or unsupported automatic restart")
    if os.environ["SLURMD_NODENAME"] != hosts[rank]:
        raise ValueError("Slurm node and rank disagree")
    spec = json.loads((control / "storage-spec.json").read_text())
    baseline = json.loads(Path(spec["baseline_study"]).read_text())
    if (baseline["trainer_gpus"], baseline["inference_gpus"], baseline["lag"], baseline["max_steps"]) != (
        4,
        8,
        256,
        1000,
    ):
        raise ValueError("Expected the authorized four-trainer/eight-inference exact256 full study")
    allocation = layout(hosts, baseline["inference_port"])
    shared = control / "attempts" / os.environ["SLURM_JOB_ID"]
    node_control = shared / hosts[rank]
    node_control.mkdir(parents=True, exist_ok=False)
    for name in ("local_backup.py", "node_local_run.py", "launch_full_run.py", "probe_allocated_gpus.py"):
        shutil.copy2(control / name, node_control / name)
    workspace, runtime = Path(spec["workspace"]), Path(spec["runtime"])
    processes, streams = {}, []
    backup = None
    interrupted = False
    failure = None
    completed = False
    output = workspace / "runs" / spec["run_name"]

    def interrupt(*_):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    def guard():
        if interrupted:
            raise RuntimeError("Multi-node supervisor received termination")
        failures = list(shared.glob("*/failure.json"))
        if failures:
            raise RuntimeError(f"A node reported failure: {failures[0]}")
        for name, process in processes.items():
            if process.poll() is not None:
                if name == "training" and process.returncode == 0:
                    continue
                raise RuntimeError(f"{name} exited with code {process.returncode}")
        if backup is not None and backup.poll() is not None:
            raise RuntimeError("Backup process exited before finalization")

    def wait_until(predicate, timeout):
        started = time.monotonic()
        while not predicate():
            guard()
            if time.monotonic() - started > timeout:
                raise TimeoutError("Multi-node startup coordination timed out")
            time.sleep(2)

    def start(name, argv, env):
        stream = (workspace / f"{name}.log").open("x")
        streams.append(stream)
        process = subprocess.Popen(
            [str(x) for x in argv],
            env=env,
            cwd=workspace / "release",
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes[name] = process
        return process

    try:
        stage(spec, node_control)
        guard()
        install_router(control, runtime)
        release = workspace / "release"
        python = runtime / "prime-rl/.venv/bin/python"
        environment = local_environment(runtime, workspace, release)
        environment.update(NCCL_P2P_DISABLE="1", NCCL_SHM_DISABLE="1")
        command(
            [
                python,
                control / "probe_allocated_gpus.py",
                "--output",
                workspace / "gpu-probes.json",
                "--timeout",
                "180",
            ],
            env=environment,
            timeout=240,
        )
        probes = json.loads((workspace / "gpu-probes.json").read_text())
        devices = validate_health(probes)
        command(
            [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=6",
                release / "scripts/gpu_health.py",
                "--expected-gpus",
                "6",
                "--receipt",
                workspace / "collective.json",
            ],
            env=environment,
            timeout=600,
        )
        collective = json.loads((workspace / "collective.json").read_text())
        if collective["world_size"] != 6 or len(collective["devices"]) != 6:
            raise ValueError("Collective test did not cover the full local allocation")
        atomic_json(
            node_control / "health.json",
            {"probes": probes, "collective": collective, "ip": socket.gethostbyname(hosts[rank])},
        )
        wait_until(lambda: all((shared / host / "health.json").is_file() for host in hosts), 7200)
        receipts = [json.loads((shared / host / "health.json").read_text()) for host in hosts]
        uuids = [row["uuid"] for receipt in receipts for row in receipt["probes"]["results"]]
        if len(set(uuids)) != 12:
            raise ValueError("The allocation does not contain twelve distinct GPUs")
        command(
            [
                python,
                "-m",
                "torch.distributed.run",
                "--nnodes=2",
                "--nproc-per-node=6",
                f"--node-rank={rank}",
                f"--master-addr={hosts[0]}",
                f"--master-port={baseline['inference_port'] + 50}",
                release / "scripts/multinode_health.py",
                "--receipt",
                workspace / "network-health.json",
            ],
            env=environment,
            timeout=600,
        )
        if rank == 0:
            shutil.copy2(workspace / "network-health.json", node_control / "network-health.json")
        atomic_json(workspace / "remote-inference.json", {**allocation, "hardware": receipts})
        command(
            [
                python,
                "-m",
                "deepseek_study.cli",
                "build",
                workspace / "study.json",
                workspace / "configs",
            ],
            env=environment,
            timeout=180,
        )
        base = json.loads((workspace / "configs/inference.json").read_text())
        count = 2 if rank == 0 else 6
        for index in range(count):
            config = json.loads(json.dumps(base))
            config["router"] = None
            config["server"].update(host="0.0.0.0", port=baseline["inference_port"] + 100 + index)
            config["vllm"].update(
                data_parallel_size=1,
                data_parallel_size_local=1,
                api_server_count=1,
                seed=worker_seed(baseline["seed"], rank, index),
            )
            config["output_dir"] = str(workspace / f"inference-{index}")
            config_path = workspace / f"inference-{index}.json"
            atomic_json(config_path, config)
            atomic_json(node_control / config_path.name, config)
            worker_env = {**environment, "CUDA_VISIBLE_DEVICES": devices[index]}
            for key in ("VLLM_RPC_BASE_PATH", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT"):
                worker_env[key] = environment[key] + f"/worker-{index}"
                Path(worker_env[key]).mkdir(parents=True, exist_ok=True)
            start(f"inference-{index}", [python, "-m", "prime_rl.entrypoints.inference", "@", config_path], worker_env)
        if rank == 0:
            start(
                "router",
                [
                    python,
                    "-m",
                    "vllm_router.launch_router",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(baseline["inference_port"]),
                    "--policy",
                    "power_of_two",
                    "--worker-urls",
                    *[url.removesuffix("/v1") for url in allocation["worker_urls"]],
                    "--worker-startup-timeout-secs",
                    "1800",
                    "--retry-max-retries",
                    "1",
                    "--request-timeout-secs",
                    "86400",
                    "--request-id-headers",
                    "x-session-id",
                ],
                {**environment, "CUDA_VISIBLE_DEVICES": ""},
            )
            backup_log = (workspace / "backup.log").open("x")
            streams.append(backup_log)
            backup = subprocess.Popen(
                [
                    str(python),
                    str(control / "local_backup.py"),
                    "--source",
                    str(output),
                    spec["backup"],
                    "--metrics",
                    spec["metrics"],
                    "--stop-file",
                    str(workspace / "backup.stop"),
                    "--status",
                    str(workspace / "backup-status.json"),
                    "--lock",
                    str(runtime.parent / "backup-transfer.lock"),
                ],
                env=environment,
                stdout=backup_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            training = start(
                "training",
                [python, "-m", "deepseek_study.cli", "run", workspace / "study.json"],
                {
                    **environment,
                    "CUDA_VISIBLE_DEVICES": ",".join(devices[2:]),
                    "DEEPSEEK_STUDY_REMOTE_INFERENCE": str(workspace / "remote-inference.json"),
                },
            )
            started = time.monotonic()
            first_update = False
            while training.poll() is None:
                guard()
                if shutil.disk_usage(workspace).free < 50 * 1024**3:
                    raise RuntimeError("Local disk reserve fell below 50 GiB")
                if output.is_dir():
                    storage = output / "storage"
                    storage.mkdir(exist_ok=True)
                    if (workspace / "backup-status.json").is_file():
                        atomic_json(
                            storage / "backup-status.json", json.loads((workspace / "backup-status.json").read_text())
                        )
                first_update = first_update or (output / "updates.jsonl").exists()
                if not first_update and time.monotonic() - started > 7200:
                    raise TimeoutError("No first committed update within two hours")
                time.sleep(3)
            if training.returncode != 0 or not (output / "study-complete.json").is_file():
                raise RuntimeError("Training exited without successful completion")
            completed = True
            atomic_json(shared / "done.json", {"completed": True})
        else:
            while not (shared / "done.json").exists():
                guard()
                if shutil.disk_usage(workspace).free < 50 * 1024**3:
                    raise RuntimeError("Inference node local disk reserve fell below 50 GiB")
                time.sleep(3)
            completed = json.loads((shared / "done.json").read_text()).get("completed", False)
    except BaseException as error:
        failure = repr(error)
        atomic_json(node_control / "failure.json", {"error": failure, "time": time.time()})
        print(failure, flush=True)
    finally:
        for name, process in list(processes.items()):
            if name == "training":
                stop_process(process, 180)
        if rank == 0:
            atomic_json(shared / "done.json", {"completed": completed, "error": failure})
        for name, process in processes.items():
            if name != "training":
                stop_process(process)
        if backup is not None:
            (workspace / "backup.stop").touch()
            try:
                backup.wait(timeout=3600)
                if backup.returncode:
                    failure = failure or "Final shared backup failed"
            except subprocess.TimeoutExpired:
                stop_process(backup)
                failure = failure or "Final shared backup timed out; local outputs retained"
        if workspace.exists():
            for path in workspace.glob("*.log"):
                shutil.copy2(path, node_control / path.name)
        atomic_json(
            node_control / "result.json",
            {
                "completed": completed,
                "error": failure,
                "backup_exit": backup.poll() if backup else None,
                "local_files_retained": True,
            },
        )
        for stream in streams:
            stream.close()
    if failure or not completed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
