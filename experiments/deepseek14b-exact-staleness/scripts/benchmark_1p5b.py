import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def run(command, log, environment, timeout):
    with log.open("x") as stream:
        process = subprocess.Popen(
            command, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = process.wait(timeout=timeout)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    if code:
        raise RuntimeError(f"Benchmark subprocess exited {code}: {log}")


def stage(args):
    from node_local_run import copy_tree, digest, local_environment, relocate_runtime, secure_local

    workspace = secure_local(args.workspace)
    if (workspace / "ready.json").exists():
        raise FileExistsError("Refusing to reuse a prepared benchmark workspace")
    if shutil.disk_usage(workspace).free < 60 * 1024**3:
        raise RuntimeError("Benchmark needs 60 GiB of free node-local storage")
    release = workspace / "release"
    for name, expected in json.loads((args.release / "PACKAGE_SHA256.json").read_text()).items():
        if digest(args.release / name) != expected:
            raise ValueError(f"Frozen release changed: {name}")
    copy_tree(args.release, release, ("vendor", "__pycache__"))
    scripts = workspace / "scripts"
    copy_tree(args.control, scripts)
    runtime = secure_local(workspace / "runtime")
    python = relocate_runtime(runtime, args.prime, release)
    environment = local_environment(runtime, workspace, release)
    study = json.loads(args.study.read_text())
    model = workspace / "model"
    model.mkdir()
    for source in Path(study["prepared_model_path"]).rglob("*"):
        if source.is_file():
            target = model / source.relative_to(study["prepared_model_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if digest(source) != digest(target):
                raise ValueError("Staged model or tokenizer differs")
    original = sorted((args.production / "rollouts").glob("0-warmup-*.msgpack"))
    if len(original) != 1:
        raise ValueError("Expected one immutable policy-zero rollout archive")
    shutil.copy2(original[0], workspace / "archive.msgpack")
    if digest(original[0]) != digest(workspace / "archive.msgpack"):
        raise ValueError("Staged rollout archive differs")
    configs = workspace / "configs"
    configs.mkdir()
    for name in ("trainer", "orchestrator"):
        data = json.loads((args.production / "configs" / f"{name}.json").read_text())
        data["model"]["name"] = str(model)
        data["tokenizer"]["name"] = str(model)
        (configs / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    (workspace / "ready.json").write_text(
        json.dumps({"source_archive_sha256": digest(original[0]), "python": str(python)})
    )
    os.execve(
        python,
        [
            str(python),
            str(scripts / "benchmark_1p5b.py"),
            "matrix",
            "--workspace",
            str(workspace),
            "--shared",
            str(args.shared),
        ],
        environment,
    )


def matrix(args):
    import msgspec
    from prime_rl.configs.orchestrator import OrchestratorConfig
    from prime_rl.orchestrator.packing import BatchPacker
    from prime_rl.transports.batch import TrainingSample
    from prime_rl.utils.process import DEFAULT_COMMON_ENV_VARS, DEFAULT_INFERENCE_ENV_VARS, DEFAULT_TRAINER_ENV_VARS

    workspace = args.workspace
    os.chdir(workspace)
    scripts = workspace / "scripts"
    results = workspace / ("results-" + os.environ["SLURM_JOB_ID"])
    results.mkdir(exist_ok=False)
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices) != 8 or len(set(devices)) != 8:
        raise RuntimeError("Benchmark requires the full verified eight-GPU allocation")
    environment = {**os.environ, **DEFAULT_COMMON_ENV_VARS, **DEFAULT_TRAINER_ENV_VARS}
    environment.update(
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NCCL_P2P_DISABLE="1", NCCL_SHM_DISABLE="1"
    )
    launch = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8"]
    run(
        launch + [str(scripts / "gpu_health.py"), "--expected-gpus", "8", "--receipt", str(results / "health.json")],
        results / "health.log",
        environment,
        600,
    )
    archive = msgspec.msgpack.decode((workspace / "archive.msgpack").read_bytes())
    if archive["format"] != 2 or archive["behavior_version"] != 0:
        raise ValueError("Replay archive must be the immutable policy-zero version-two cohort")
    samples = msgspec.msgpack.decode(archive["payload"]["samples"], type=list[TrainingSample])
    if len(samples) != 512:
        raise ValueError("Replay requires the complete original 512 responses")
    prompts = []
    for i, sample in enumerate(samples):
        start = sample.mask.index(True)
        if start < 1 or start > 2048 or not all(sample.mask[start:]):
            raise ValueError("Expected a single-turn prompt followed by sampled response tokens")
        prompts.append({"index": i, "seed": 42 + i, "prompt_token_ids": sample.token_ids[:start]})
    prompt_path = workspace / "prompts.json"
    prompt_path.write_text(json.dumps(prompts))
    config = OrchestratorConfig.model_validate_json((workspace / "configs/orchestrator.json").read_text())
    config.num_train_workers = 4
    grid = BatchPacker(config).pack(samples)
    grid_path = workspace / "grid.msgpack"
    grid_path.write_bytes(msgspec.msgpack.encode(grid))
    (results / "workload.json").write_text(
        json.dumps(
            {
                "prompts_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
                "grid_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
                "rank_micro_batches": [len(rank) for rank in grid],
                "responses": 512,
                "trainer_devices": devices[4:],
                "inference_devices": devices[:4],
                "production_inference_gpus": 3,
                "benchmark_inference_gpus": 4,
            },
            indent=2,
        )
    )
    statuses = {}
    publication_lock = threading.Lock()
    sampling_stop = threading.Event()

    def sample_gpus():
        fields = "uuid,name,memory.total,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw"
        with (results / "gpu-samples.jsonl").open("x") as stream:
            while not sampling_stop.is_set():
                record = {"time": time.time(), "fields": fields.split(",")}
                try:
                    sample = subprocess.run(
                        ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=True,
                    )
                    record["devices"] = sample.stdout.splitlines()
                except Exception as error:
                    record["error"] = str(error)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                sampling_stop.wait(5)

    def inference():
        for label, limit in (("baseline", 16), ("seq32", 32), ("seq64", 64), ("baseline-repeat", 16)):
            started = time.time()
            folder = results / ("inference-" + label)
            folder.mkdir()
            errors = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = []
                for shard, device in enumerate(devices[:4]):
                    env = {
                        **environment,
                        **DEFAULT_INFERENCE_ENV_VARS,
                        "CUDA_VISIBLE_DEVICES": device,
                        "VLLM_CACHE_ROOT": str(workspace / "cache" / f"vllm-{label}-{shard}"),
                        "TRITON_CACHE_DIR": str(workspace / "cache" / f"triton-{label}-{shard}"),
                        "TORCHINDUCTOR_CACHE_DIR": str(workspace / "cache" / f"inductor-{label}-{shard}"),
                    }
                    command = [
                        sys.executable,
                        str(scripts / "benchmark_inference.py"),
                        "--model",
                        str(workspace / "model"),
                        "--prompts",
                        str(prompt_path),
                        "--output",
                        str(folder / f"shard-{shard}.json"),
                        "--shard",
                        str(shard),
                        "--sequences",
                        str(limit),
                    ]
                    futures.append(pool.submit(run, command, folder / f"shard-{shard}.log", env, 1800))
                for future in futures:
                    try:
                        future.result()
                    except Exception as error:
                        errors.append(str(error))
            statuses["inference-" + label] = {"seconds_including_startup": time.time() - started, "errors": errors}
            publish()
            if label == "baseline" and errors:
                return

    def trainer():
        cases = [
            ("baseline", {}),
            ("baseline-repeat", {}),
            ("no-ac", {"ac": None}),
            ("no-reshard", {"reshard_after_forward": False}),
            ("no-ac-no-reshard", {"ac": None, "reshard_after_forward": False}),
            ("compile", {"compile": {}}),
        ]
        for label, changes in cases:
            started = time.time()
            override = results / ("overrides-" + label + ".json")
            override.write_text(json.dumps(changes))
            env = {
                **environment,
                "CUDA_VISIBLE_DEVICES": ",".join(devices[4:]),
                "TORCHINDUCTOR_CACHE_DIR": str(workspace / "cache" / ("trainer-" + label)),
                "TRITON_CACHE_DIR": str(workspace / "cache" / ("trainer-triton-" + label)),
            }
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=4",
                str(scripts / "profile_trainer.py"),
                "--config",
                str(workspace / "configs/trainer.json"),
                "--grid",
                str(grid_path),
                "--output",
                str(results / ("trainer-" + label)),
                "--micro-batches",
                "8",
                "--steps",
                "3",
                "--model-overrides",
                str(override),
            ]
            error = None
            try:
                run(command, results / ("trainer-" + label + ".log"), env, 1800)
            except Exception as exception:
                error = str(exception)
            statuses["trainer-" + label] = {"seconds_including_startup": time.time() - started, "error": error}
            publish()
            if label == "baseline" and error:
                return

    def publish():
        with publication_lock:
            args.shared.mkdir(parents=True, exist_ok=True)
            (results / "status.json").write_text(json.dumps(dict(statuses), indent=2))
            subprocess.run(["rsync", "-rt", str(results) + "/", str(args.shared) + "/"], check=True, timeout=180)

    sampler = threading.Thread(target=sample_gpus, daemon=True)
    sampler.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(inference), pool.submit(trainer)]
            for future in futures:
                future.result()
    finally:
        sampling_stop.set()
        sampler.join(timeout=15)
        (results / "status.json").write_text(json.dumps(statuses, indent=2))
        publish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("stage", "matrix"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--prime", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--production", type=Path)
    args = parser.parse_args()
    if args.phase == "stage":
        if any(getattr(args, field) is None for field in ("control", "prime", "release", "study", "production")):
            parser.error("Staging requires control, prime, release, study and production paths")
        stage(args)
    else:
        matrix(args)


if __name__ == "__main__":
    main()
