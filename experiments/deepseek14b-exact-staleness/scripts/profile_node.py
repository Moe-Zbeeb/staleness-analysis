import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--single-group", action="store_true")
    parser.add_argument("--peer-protocol", choices=("auto", "Simple"), default="auto")
    parser.add_argument("--peer-ctas", type=int)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.peer_ctas is not None and not 1 <= args.peer_ctas <= 32:
        raise ValueError("Profiling supports between one and 32 communication CTAs")
    from prime_rl.utils.process import DEFAULT_COMMON_ENV_VARS, DEFAULT_TRAINER_ENV_VARS

    scripts = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=False)
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    count = args.expected_gpus
    if len(devices) != count or len(set(devices)) != count or (not args.single_group and count != 8):
        raise ValueError("Profiling must cover the complete allocation")
    group_size = count if args.single_group else 4
    environment = os.environ.copy()
    environment.update(DEFAULT_COMMON_ENV_VARS)
    environment.update(DEFAULT_TRAINER_ENV_VARS)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    for key in ("NCCL_PROTO", "NCCL_MIN_CTAS", "NCCL_MAX_CTAS", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS"):
        environment.pop(key, None)
    peer_overrides = {}
    if args.peer_protocol != "auto":
        peer_overrides["NCCL_PROTO"] = args.peer_protocol
    if args.peer_ctas is not None:
        peer_overrides.update(NCCL_MIN_CTAS=str(args.peer_ctas), NCCL_MAX_CTAS=str(args.peer_ctas))
    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone"]

    def run(name, command, env=None, timeout=900):
        print(json.dumps({"starting": name, "command": command}), flush=True)
        with (args.output / f"{name}.log").open("x") as stream:
            subprocess.run(
                command, env=env or environment, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=True
            )

    run("topology", ["nvidia-smi", "topo", "-m"])
    run("hardware", ["nvidia-smi", "-q"])
    run(
        "health",
        launcher
        + [
            f"--nproc-per-node={count}",
            str(scripts / "gpu_health.py"),
            "--expected-gpus",
            str(count),
            "--receipt",
            str(args.output / "health.json"),
        ],
    )
    for transport, value in (("network", "1"), ("peer", "0")):
        env = {**environment, "NCCL_P2P_DISABLE": value, "NCCL_SHM_DISABLE": value, "NCCL_DEBUG": "INFO"}
        if transport == "peer":
            env.update(peer_overrides)
        run(
            f"collectives-{transport}",
            launcher
            + [
                f"--nproc-per-node={count}",
                str(scripts / "profile_collectives.py"),
                "--transport",
                transport,
                "--output",
                str(args.output / "collectives"),
                "--group-size",
                str(group_size),
            ],
            env,
        )
    health = json.loads((args.output / "health.json").read_text())
    if not args.single_group and min(device["bytes"] for device in health["devices"]) < 79 * 1024**3:
        raise RuntimeError("Collective profiles complete; four-rank model replay requires 80GB GPUs")
    import msgspec
    from prime_rl.configs.orchestrator import OrchestratorConfig
    from prime_rl.orchestrator.packing import BatchPacker
    from prime_rl.transports.batch import TrainingSample

    source = sorted((args.production / "rollouts").glob("0-warmup-*.msgpack"))
    if len(source) != 1:
        raise ValueError("Expected one immutable policy-zero warmup archive")
    archive = msgspec.msgpack.decode(source[0].read_bytes())
    if archive["format"] != 2 or archive["behavior_version"] != 0:
        raise ValueError("Replay requires a version-two policy-zero archive")
    samples = msgspec.msgpack.decode(archive["payload"]["samples"], type=list[TrainingSample])
    orchestrator = OrchestratorConfig.model_validate_json(
        (args.production / "configs" / "orchestrator.json").read_text()
    )
    orchestrator.num_train_workers = group_size
    grid = BatchPacker(orchestrator).pack(samples)
    grid_path = args.output / "replay.msgpack"
    grid_path.write_bytes(msgspec.msgpack.encode(grid))
    (args.output / "replay-source.json").write_text(
        json.dumps(
            {
                "archive": str(source[0]),
                "rank_micro_batches": [len(rank) for rank in grid],
                "samples": len(samples),
                "profiling_world_size": group_size,
                "production_world_size": 4,
                "single_group": args.single_group,
                "peer_overrides": peer_overrides,
                "trainer_defaults": DEFAULT_TRAINER_ENV_VARS,
                "common_defaults": DEFAULT_COMMON_ENV_VARS,
                "trace": args.trace,
            },
            indent=2,
        )
        + "\n"
    )
    for round_index in range(2):
        processes = []
        try:
            for group in range(1 if args.single_group else 2):
                transport = "network" if group == round_index else "peer"
                value = "1" if transport == "network" else "0"
                name = f"replay-{round_index}-{transport}"
                env = {
                    **environment,
                    "CUDA_VISIBLE_DEVICES": ",".join(devices[group * group_size : (group + 1) * group_size]),
                    "NCCL_P2P_DISABLE": value,
                    "NCCL_SHM_DISABLE": value,
                }
                if transport == "peer":
                    env.update(peer_overrides)
                command = launcher + [
                    f"--nproc-per-node={group_size}",
                    str(scripts / "profile_trainer.py"),
                    "--config",
                    str(args.production / "configs" / "trainer.json"),
                    "--grid",
                    str(grid_path),
                    "--output",
                    str(args.output / name),
                ]
                if args.trace and round_index == 0:
                    command.append("--trace")
                stream = (args.output / f"{name}.log").open("x")
                print(json.dumps({"starting": name, "devices": env["CUDA_VISIBLE_DEVICES"]}), flush=True)
                process = subprocess.Popen(
                    command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
                )
                processes.append((process, stream, name))
            for process, stream, name in processes:
                if process.wait(timeout=2400) != 0:
                    raise RuntimeError(f"Bounded replay failed: {name}")
        finally:
            import signal

            for process, stream, name in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                stream.close()
    print(json.dumps({"status": "completed", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
