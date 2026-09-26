import json
import importlib
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from deepseek_study import PRIME_COMMIT
from deepseek_study.runtime import checkpoints
from deepseek_study.dataset.assets import validate_prepared
from deepseek_study.runtime.build import build
from deepseek_study.runtime.identity import capture, snapshot
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


def launch(study, root, resume=None):
    verify_upstream(root)
    configure_nccl_transport()
    preflight = validate_prepared(study)
    identity = capture(root, study)
    starting_step = 0
    if resume:
        directory = Path(resume).resolve()
        checkpoints.verify_components(directory)
        state = checkpoints.load(directory / "study", study.fingerprint(), identity["sha256"])
        starting_step = state.completed_steps
        if state.completed_steps >= study.max_steps:
            raise ValueError("Checkpoint already completed the configured training budget")
        if (
            not (directory / "trainer" / ".metadata").is_file()
            or not (directory / "orchestrator" / "progress.pt").is_file()
        ):
            raise ValueError("Checkpoint is missing trainer or sampler state")
    import torch

    count = torch.cuda.device_count()
    total = study.trainer_gpus + study.inference_gpus
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
    config = build(study, config_dir, resume)
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
                "starting_step": starting_step,
                "resume_from": str(Path(resume).resolve()) if resume else None,
                "config_sha256": study.fingerprint(),
                "identity_sha256": identity["sha256"],
                "hardware": hardware,
                "nccl_transport": {
                    key: os.environ.get(key) for key in ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE")
                },
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
        "PYTHONHASHSEED": str(study.seed),
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": str(output / "source" / "src")
        + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
    }
    processes = {}
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)

    def start(name, args, env):
        with (logs / f"{name}.log").open("w") as stream:
            processes[name] = subprocess.Popen(
                args, env={**base_env, **env}, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )

    run_status = "crashed"
    observer_warned = False
    paper_warned = False
    try:
        checkpoints.atomic_write(output / "paper-observer.json", b'{"enabled":true}')
        start(
            "paper-metrics",
            [sys.executable, "-m", "deepseek_study.tracking.observer", str(output), "--parent-pid", str(os.getpid())],
            {"CUDA_VISIBLE_DEVICES": "", "RANK": "0", "LOCAL_RANK": "0"},
        )
        if os.environ.get("DEEPSEEK_STUDY_RUNBOARD", "1") != "0":
            try:
                start(
                    "runboard",
                    [
                        sys.executable,
                        "-m",
                        "deepseek_study.tracking.runboard",
                        str(output),
                        "--parent-pid",
                        str(os.getpid()),
                    ],
                    {"CUDA_VISIBLE_DEVICES": "", "RANK": "0", "LOCAL_RANK": "0"},
                )
            except Exception as error:
                print(
                    f"Runboard observer could not start ({type(error).__name__}); training continues", file=sys.stderr
                )
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
            {**DEFAULT_TRAINER_ENV_VARS, "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids[study.inference_gpus :])},
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
                if name == "runboard":
                    if code is not None and not observer_warned:
                        print("Runboard observer exited; training continues. See logs/runboard.log", file=sys.stderr)
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
    except KeyboardInterrupt:
        run_status = "killed"
        raise
    finally:
        observer = processes.pop("runboard", None)
        paper_observer = processes.pop("paper-metrics", None)
        for process in processes.values():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 20
        for process in processes.values():
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for process in processes.values():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        try:
            checkpoints.atomic_write(output / "run-status.json", json.dumps({"status": run_status}).encode())
        except Exception as error:
            print(f"Could not record launcher status ({type(error).__name__})", file=sys.stderr)
        if paper_observer is not None:
            try:
                paper_observer.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(paper_observer.pid, signal.SIGKILL)
                paper_observer.wait()
                checkpoints.atomic_write(
                    output / "paper-status.json",
                    b'{"status":"interrupted","error":"Replay with paper-metrics --once"}',
                )
                print("Paper metrics need offline replay after the shutdown deadline", file=sys.stderr)
        if observer is not None:
            try:
                observer.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(observer.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                observer.wait()
                print("Runboard observer exceeded shutdown deadline; original logs remain available", file=sys.stderr)
        signal.signal(signal.SIGTERM, previous_term)
    return output
