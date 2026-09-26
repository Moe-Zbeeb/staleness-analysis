import asyncio
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from .common import digest, immutable_json, io_slot, lock, read, write
from .discovery import discover
from .export import export_model
from .queue import LostLease, Queue
from .runner import WorkerProfile, evaluate


def execute(root, claim_path):
    queue, claim = Queue(root), read(claim_path)
    queue.progress(claim)
    spec = claim["spec"]
    if spec["kind"] == "export":
        with io_slot(queue.root):
            receipt = export_model(spec["source"], spec["model_source"], spec["model"],
                                   progress=lambda: queue.progress(claim))
        queue.finish(claim, receipt)
    else:
        profile = WorkerProfile.model_validate(read(queue.root / "profile.json"))
        asyncio.run(evaluate(queue, claim, profile))


def terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # vLLM may leave engine children after its Python parent exits. All children
    # were started in this attempt's process group and must release their GPUs.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def work(root, kind, once=False, stall_seconds=1800, max_tasks=100):
    queue = Queue(root)
    owner = f"{socket.gethostname()}:{os.environ.get('SLURM_JOB_ID', 'local')}:{os.getpid()}"
    completed = 0
    while completed < max_tasks:
        claim = queue.claim(kind, owner)
        if claim is None:
            break
        attempt = queue.root / "attempts" / claim["id"] / claim["token"]
        attempt.mkdir(parents=True)
        write(attempt / "claim.json", claim)
        with (attempt / "worker.log").open("w") as output:
            process = subprocess.Popen(
                [sys.executable, "-m", "deepseek_study.evaluation", "execute", str(queue.root),
                 str(attempt / "claim.json")], stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
            )
            error = None
            try:
                while process.poll() is None:
                    try:
                        last_progress = queue.heartbeat(claim)
                    except LostLease:
                        if queue.snapshot()["tasks"][claim["id"]]["state"] == "done":
                            process.wait(timeout=30)
                            break
                        raise
                    if time.time() - last_progress > stall_seconds:
                        raise TimeoutError(f"No durable progress for {stall_seconds}s; see {attempt / 'worker.log'}")
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        pass
                if process.returncode:
                    raise RuntimeError(f"Worker exited {process.returncode}; see {attempt / 'worker.log'}")
                if queue.snapshot()["tasks"][claim["id"]]["state"] != "done":
                    raise RuntimeError("Worker exited without a completion receipt")
            except BaseException as failure:
                error = failure
                terminate(process)
                try:
                    queue.fail(claim, failure)
                except LostLease:
                    pass
            finally:
                terminate(process)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise error
        completed += 1
        if once:
            break
    return {"processed_attempts": completed}


def initialize(root, profile):
    root = Path(root).resolve()
    immutable_json(root / "profile.json", profile.model_dump())
    Queue(root)
    return root


def slurm_command(root, kind, python, uv, account, partition, qos, excluded_nodes, cpus=8):
    root = Path(root).resolve()
    profile = WorkerProfile.model_validate(read(root / "profile.json"))
    name = f"dse-{digest(str(root))[:10]}-{kind}"
    logs = root / "slurm"
    logs.mkdir(exist_ok=True)
    command = ["sbatch", "--parsable", f"--job-name={name}", f"--account={account}",
               f"--partition={partition}", f"--qos={qos}", "--nodes=1", "--ntasks=1",
               f"--cpus-per-task={cpus}", f"--mem={'32G' if kind == 'export' else '64G'}",
               "--time=12:00:00", "--signal=B:TERM@90", "--no-requeue",
               f"--output={logs}/%j.log"]
    if kind == "evaluate":
        if not excluded_nodes:
            raise ValueError("Explicit exclusions restricting placement to verified A100 40GB nodes required")
        command += [f"--gres=gpu:a100:{profile.tensor_parallel}", f"--exclude={excluded_nodes}"]
    source = str(Path(__file__).resolve().parents[2])
    body = (
        "set -eu\n"  # sbatch --wrap uses /bin/sh, which may be dash.
        f"export PYTHONPATH={shlex.quote(source)}\n"
        f"export OMP_NUM_THREADS={cpus}\n"
        "export PYTHONUNBUFFERED=1\n"
        "export VLLM_WORKER_MULTIPROC_METHOD=spawn\n"
        "export TOKENIZERS_PARALLELISM=false\n"
        "export HF_HUB_OFFLINE=1\n"
        "export TRANSFORMERS_OFFLINE=1\n"
        + "exec " + shlex.join([uv, "run", "--no-project", python, "-m", "deepseek_study.evaluation", "work",
                                str(root), "--kind", kind]) + "\n"
    )
    return name, command + ["--wrap", body]


def reconcile_slurm(root, limits, nodes, **options):
    submitted = []
    state = Queue(root).snapshot()
    allowed = set(subprocess.run(["scontrol", "show", "hostnames", nodes], check=True, text=True,
                                 capture_output=True, timeout=30).stdout.split())
    inventory = set(subprocess.run(["sinfo", "--Node", "--noheader", "--format=%N"], check=True, text=True,
                                   capture_output=True, timeout=30).stdout.split())
    if not allowed or not allowed <= inventory:
        raise ValueError("Configured evaluator nodes absent from the cluster inventory")
    # --nodelist requests ALL named nodes. Its complement in --exclude lets
    # Slurm select ANY one eligible node, including when every GPU is busy now.
    excluded = ",".join(sorted(inventory - allowed))
    for kind, limit in limits.items():
        ready = sum(t["state"] == "queued" and t["spec"]["kind"] == kind and
                    (not t["depends_on"] or state["tasks"][t["depends_on"]]["state"] == "done")
                    for t in state["tasks"].values())
        name, command = slurm_command(root, kind, excluded_nodes=excluded, **options)
        output = subprocess.run(["squeue", "--me", "--name", name, "--noheader", "--format=%i"],
                                check=True, text=True, capture_output=True, timeout=30)
        active = len(output.stdout.split())
        for _ in range(min(ready, max(0, limit - active))):
            # Do not auto-repeat an ambiguous submission; surface it and reconcile
            # scheduler state on the next invocation before considering a new job.
            result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=60)
            job = result.stdout.strip().split(";")[0]
            if not job.isdigit():
                raise ValueError(f"Unrecognized sbatch receipt: {result.stdout}")
            receipt = {"job": job, "kind": kind, "time": time.time(), "command": command}
            write(Path(root) / "submissions" / f"{job}.json", receipt)
            submitted.append(receipt)
    return submitted


def serve(root, once=False, interval=60, slurm=None):
    root = Path(root)
    with lock(root / "coordinator.lock", blocking=False):
        while True:
            discovery = discover(root)
            status = {"time": time.time(), "discovery": discovery, "queue": Queue(root).snapshot()}
            if slurm:
                try:
                    status["submissions"] = reconcile_slurm(root, **slurm)
                except (subprocess.SubprocessError, OSError) as error:
                    status["scheduler_error"] = {"type": type(error).__name__, "error": str(error)}
            write(root / "status.json", status)
            from .report import collect

            write(root / "metrics.json", collect([root]))
            if once:
                return status
            time.sleep(interval)
