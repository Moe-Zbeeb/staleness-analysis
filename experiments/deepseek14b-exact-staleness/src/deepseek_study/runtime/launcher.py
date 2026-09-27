import json
import importlib
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from deepseek_study import MODEL_ID, MODEL_REVISION, PRIME_COMMIT
from deepseek_study.runtime import checkpoints
from deepseek_study.dataset.assets import validate_prepared
from deepseek_study.runtime.build import build
from deepseek_study.runtime.identity import capture, snapshot
from deepseek_study.runtime.processes import drain_process, stop_process_groups
from deepseek_study.tracking.archive import reserve_mirror
from prime_rl.entrypoints.rl import env_servers
from prime_rl.utils.process import DEFAULT_COMMON_ENV_VARS, DEFAULT_INFERENCE_ENV_VARS, DEFAULT_TRAINER_ENV_VARS


def verify_upstream(root):
    vendor = root / "vendor" / "prime-rl"

    def git(*args):
        return subprocess.check_output(["git", "-C", str(vendor), *args], text=True).strip()

    if git("rev-parse", "HEAD") != PRIME_COMMIT:
        raise ValueError("Official PrimeRL checkout is not at the pinned commit")
    if git("diff", "HEAD", "--", "."):
        raise ValueError("Official PrimeRL or a submodule contains tracked changes")
    for line in git("submodule", "status").splitlines():
        if line.startswith("+") or (line.startswith("-") and "prime-kernels" not in line):
            raise ValueError("An official dependency is missing or at a different commit")
    for name in ("prime_rl.orchestrator", "prime_rl.configs", "verifiers.v1", "renderers", "pydantic_config"):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(vendor.resolve()):
            raise ValueError(f"Python imported {name} from another installation")


def configure_nccl_transport():
    from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable

    disable_nccl_p2p_if_unavailable()


def resume_step(directory, study, identity_hash):
    checkpoints.verify_components(directory)
    state = checkpoints.load(directory / "study", study.fingerprint(), identity_hash)
    if state.completed_steps >= study.max_steps:
        raise ValueError("Checkpoint already completed the configured training budget")
    if (
        not (directory / "trainer" / ".metadata").is_file()
        or not (directory / "orchestrator" / "progress.pt").is_file()
    ):
        raise ValueError("Checkpoint is missing trainer or sampler state")
    return state.completed_steps


def launch(study, root, resume=None):
    from deepseek_study.runtime.deployment import RemoteInference

    remote_path = os.environ.get("DEEPSEEK_STUDY_REMOTE_INFERENCE")
    remote = RemoteInference.read(remote_path) if remote_path else None
    if remote is not None:
        remote.validate_study(study)
        if resume:
            raise ValueError("Multi-node recovery requires an explicitly validated recovery launch")
    verify_upstream(root)
    configure_nccl_transport()
    preflight = validate_prepared(study)
    identity = capture(root, study)
    starting_step = 0
    if resume:
        directory = Path(resume).resolve()
        starting_step = resume_step(directory, study, identity["sha256"])
    import torch

    count = torch.cuda.device_count()
    total = study.trainer_gpus if remote else study.trainer_gpus + study.inference_gpus
    if not torch.cuda.is_available() or count != total:
        raise ValueError(f"Allocate exactly {total} visible GPUs for this configuration; found {count}")
    if identity["runtime"]["vllm"] is None:
        raise RuntimeError(
            "GPU dependencies are missing; run scripts/bootstrap.py --gpu with the correct attention backend"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = visible.split(",") if visible else [str(index) for index in range(count)]
    if len(gpu_ids) != total or len(set(gpu_ids)) != total:
        raise ValueError("CUDA_VISIBLE_DEVICES does not describe a unique full-node allocation")
    output = study.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot(root, output / "source", identity)
    config_dir = output / "configs"
    config = build(study, config_dir, resume, remote)
    if remote is not None:
        checkpoints.atomic_write(output / "deployment.json", remote.model_dump_json(indent=2).encode())
    checkpoints.atomic_write(output / "preflight.json", json.dumps(preflight, indent=2).encode())
    hardware = [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory": torch.cuda.get_device_properties(index).total_memory,
            "capability": torch.cuda.get_device_capability(index),
        }
        for index in range(count)
    ]
    checkpoints.atomic_write(
        output / "run.json",
        json.dumps(
            {
                "run_uuid": uuid.uuid4().hex,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "tracking_backend": "tensorboard",
                "starting_step": starting_step,
                "resume_from": str(Path(resume).resolve()) if resume else None,
                "config_sha256": study.fingerprint(),
                "identity_sha256": identity["sha256"],
                "hardware": hardware,
                "nccl_transport": {key: os.environ.get(key) for key in ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE")},
            },
            indent=2,
        ).encode(),
    )
    reserve_mirror(output, study.metrics_mirror_root)
    logs = output / "logs"
    logs.mkdir()
    base_env = {
        **os.environ,
        **DEFAULT_COMMON_ENV_VARS,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        "PRL_RUN_ID": uuid.uuid4().hex,
        "PRL_RUN_NAME": output.name,
        "DEEPSEEK_STUDY_SEED": str(study.seed),
        "DEEPSEEK_STUDY_RUNBOARD": "0",
        "PYTHONHASHSEED": str(study.seed),
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": str(output / "source" / "src")
        + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
    }
    processes = {}
    previous_signals = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    received_signal = None
    received_signum = None
    registering_process = False

    def interrupted(signum, frame):
        nonlocal received_signal, received_signum
        if received_signum is None:
            received_signum = signum
            received_signal = signal.Signals(signum).name
        if registering_process:
            return
        raise KeyboardInterrupt(f"Launcher received {received_signal}")

    for signum in previous_signals:
        signal.signal(signum, interrupted)

    def start(name, args, env):
        nonlocal registering_process
        with (logs / f"{name}.log").open("w") as stream:
            registering_process = True
            try:
                processes[name] = subprocess.Popen(
                    args, env={**base_env, **env}, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
                )
            finally:
                registering_process = False
            if received_signum is not None:
                raise KeyboardInterrupt(f"Launcher received {received_signal}")

    run_status = "crashed"
    failure = None
    observer_warned = False
    paper_warned = False
    try:
        checkpoints.atomic_write(output / "paper-observer.json", b'{"enabled":true}')
        start(
            "paper-metrics",
            [sys.executable, "-m", "deepseek_study.tracking.observer", str(output), "--parent-pid", str(os.getpid())],
            {"CUDA_VISIBLE_DEVICES": "", "RANK": "0", "LOCAL_RANK": "0"},
        )
        try:
            start(
                "tensorboard",
                [
                    sys.executable,
                    "-m",
                    "deepseek_study.tracking.tensorboard",
                    str(output),
                    "--parent-pid",
                    str(os.getpid()),
                ],
                {"CUDA_VISIBLE_DEVICES": "", "RANK": "0", "LOCAL_RANK": "0"},
            )
        except Exception as error:
            print(
                f"TensorBoard observer could not start ({type(error).__name__}); training continues", file=sys.stderr
            )
        if remote is None:
            start(
                "inference",
                [sys.executable, "-m", "prime_rl.entrypoints.inference", "@", str(config_dir / "inference.json")],
                {**DEFAULT_INFERENCE_ENV_VARS, "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids[: study.inference_gpus])},
            )
        for split, source, _ in env_servers(config):
            start(
                f"env-{source.resolved_name}",
                [
                    sys.executable,
                    "-m",
                    "prime_rl.entrypoints.env_server",
                    "@",
                    str(config_dir / "envs" / split / f"{source.resolved_name}.json"),
                ],
                {"CUDA_VISIBLE_DEVICES": ""},
            )
        controller = [
            sys.executable,
            "-m",
            "deepseek_study.cli",
            "controller",
            str(config_dir / "study.json"),
            str(config_dir / "orchestrator.json"),
        ]
        if resume:
            controller += ["--resume", str(Path(resume).resolve())]
        start("controller", controller, {"CUDA_VISIBLE_DEVICES": ""})
        start(
            "trainer",
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                f"--nproc-per-node={study.trainer_gpus}",
                f"--master-port={study.inference_port + 40}",
                "--module",
                "deepseek_study.runtime.trainer",
                "@",
                str(config_dir / "trainer.json"),
            ],
            {
                **DEFAULT_TRAINER_ENV_VARS,
                "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids if remote else gpu_ids[study.inference_gpus :]),
            },
        )
        trainer_exit_deadline = None
        while True:
            status = {name: process.poll() for name, process in processes.items()}
            for name, code in status.items():
                if name == "paper-metrics":
                    if code is not None and not paper_warned:
                        print(
                            "Paper metric observer exited; raw token evidence remains. See logs/paper-metrics.log",
                            file=sys.stderr,
                        )
                        paper_warned = True
                    continue
                if name == "tensorboard":
                    if code is not None and not observer_warned:
                        print(
                            "TensorBoard observer exited; training continues. See logs/tensorboard.log",
                            file=sys.stderr,
                        )
                        observer_warned = True
                    continue
                if code is not None and (code != 0 or name not in {"controller", "trainer"}):
                    raise RuntimeError(f"{name} exited with status {code}; see {logs / (name + '.log')}")
            if status["controller"] == 0:
                if not (output / "study-complete.json").is_file():
                    raise RuntimeError("Controller exited without a successful completion marker")
                if status["trainer"] == 0:
                    run_status = "finished"
                    break
                trainer_exit_deadline = trainer_exit_deadline or time.monotonic() + study.timeout_seconds
                if time.monotonic() > trainer_exit_deadline:
                    raise RuntimeError("Trainer failed to exit after the controller completed")
            time.sleep(1)
    except KeyboardInterrupt as error:
        run_status = "killed"
        failure = {"type": type(error).__name__, "message": str(error)}
        raise
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        for signum in previous_signals:
            signal.signal(signum, signal.SIG_IGN)
        exit_codes = {name: process.poll() for name, process in processes.items()}
        observer = processes.pop("tensorboard", None)
        paper_observer = processes.pop("paper-metrics", None)
        cleanup_errors = stop_process_groups(processes)
        status_record = {
            "status": run_status,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": failure,
            "received_signal": received_signal,
            "received_signum": received_signum,
            "process_exit_codes_before_cleanup": exit_codes,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "maximum_updates": study.max_steps,
            "checkpoint_interval": study.checkpoint_interval,
            "cleanup_errors": list(cleanup_errors),
        }
        try:
            checkpoints.atomic_write(output / "run-status.json", json.dumps(status_record, indent=2).encode())
        except Exception as error:
            print(f"Could not record launcher status ({type(error).__name__})", file=sys.stderr)
        if paper_observer is not None:
            drained, errors = drain_process("paper-metrics", paper_observer)
            cleanup_errors.extend(errors)
            if not drained:
                try:
                    checkpoints.atomic_write(
                        output / "paper-status.json",
                        b'{"status":"interrupted","error":"Replay with paper-metrics --once"}',
                    )
                except Exception as error:
                    print(f"Could not record paper status ({type(error).__name__})", file=sys.stderr)
                print("Paper metrics need offline replay after the shutdown deadline", file=sys.stderr)
        if observer is not None:
            drained, errors = drain_process("tensorboard", observer)
            cleanup_errors.extend(errors)
            if not drained:
                print(
                    "TensorBoard observer exceeded shutdown deadline; replay saved metrics with track --once",
                    file=sys.stderr,
                )
        if cleanup_errors:
            for error in cleanup_errors:
                print(f"Cleanup warning: {error}", file=sys.stderr)
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
    return output
