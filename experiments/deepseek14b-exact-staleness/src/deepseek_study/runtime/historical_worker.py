import argparse
import asyncio
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from deepseek_study.config import StudyConfig
from deepseek_study.runtime.checkpoints import atomic_write


class ManualReceiver:
    async def initialize(self):
        pass

    async def sync_startup(self, step, timeout):
        if step != 0:
            raise ValueError("Historical worker must start from the pinned initial model")


class ClosedSender:
    def close(self):
        pass


def learner_alive(job_id):
    result = subprocess.run(
        ["squeue", "-j", str(job_id), "-h", "-o", "%T"], capture_output=True, text=True, timeout=20, check=False
    )
    if result.returncode:
        raise RuntimeError("Cannot verify learner scheduler state")
    states = result.stdout.split()
    return bool(states) and all(
        state in {"RUNNING", "PENDING", "CONFIGURING", "SUSPENDED", "RESIZING"} for state in states
    )


def validate_session(study, session, learner_job, identity):
    from deepseek_study.rollouts.historical import contract

    if session.get("contract") != contract(study):
        raise ValueError("Learner and historical worker contracts differ")
    if session.get("identity") != identity:
        raise ValueError("Learner and historical worker source identities differ")
    if str(session.get("learner_job_id")) != str(learner_job):
        raise ValueError("Historical session belongs to another learner allocation")
    if any(not isinstance(session.get(key), str) or not session[key] for key in ("run_uuid", "config_sha256")):
        raise ValueError("Historical session has incomplete run identity")


def validate_session_job(job, session, directory):
    from deepseek_study.rollouts.historical import worker_pool

    metadata = job["metadata"]
    if directory.name != job["job_id"]:
        raise ValueError("Historical job directory does not match its identity")
    for key in ("run_uuid", "config_sha256", "contract"):
        if metadata.get(key) != session[key]:
            raise ValueError("Historical job belongs to another learner session")
    if metadata.get("identity_sha256") != session["identity"]["sha256"]:
        raise ValueError("Historical job has a different source identity")
    version = metadata.get("policy_version")
    if type(version) is not int or not 0 <= version < session["contract"]["max_steps"] - session["contract"]["lag"]:
        raise ValueError("Historical policy version is outside the study horizon")
    if metadata.get("consumption_step") != version + session["contract"]["lag"] + 1:
        raise ValueError("Historical job has the wrong exact-staleness consumption step")
    if metadata.get("worker_pool") != worker_pool(version, session["contract"]["lag"]):
        raise ValueError("Historical job is assigned to the wrong inference pool")
    return version


def remote_versions(settings):
    return set(range(min(settings["lag"], settings["max_steps"] - settings["lag"])))


async def setup_manual_receiver(module, orchestrator):
    original = module.setup_weight_receiver
    module.setup_weight_receiver = lambda *args, **kwargs: ManualReceiver()
    try:
        await orchestrator.setup()
    finally:
        module.setup_weight_receiver = original


async def stop_orchestrator(orchestrator):
    if getattr(orchestrator, "sender", None) is None:
        orchestrator.sender = ClosedSender()
    for name in ("dispatcher", "watcher", "periodic_logger", "train_envs"):
        if not hasattr(orchestrator, name):
            setattr(orchestrator, name, None)
    await asyncio.wait_for(orchestrator.stop(), 30)


def grading_records(path, offset, response_ids):
    with path.open("rb") as stream:
        stream.seek(offset)
        records = [json.loads(line) for line in stream]
    if len(records) != len(response_ids) or {row["response_id"] for row in records} != set(response_ids):
        raise ValueError("Historical grading receipts do not match the completed cohort")
    return records


async def process_job(study, backend, directory, job, local):
    import msgspec
    from prime_rl.orchestrator.clients import update_weights

    from deepseek_study.rollouts.history_store import load_result, pin_job, publish_result, validate_job

    if job["metadata"].get("worker_pool") != "remote":
        raise ValueError("A remote worker cannot claim a local historical cohort")
    if await asyncio.to_thread(load_result, directory, job) is not None:
        return False
    with (directory / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if await asyncio.to_thread(load_result, directory, job) is not None:
            return False
        if (directory / "attempt.json").exists():
            raise RuntimeError("An incomplete historical attempt requires explicit recovery")
        atomic_write(
            directory / "attempt.json",
            json.dumps(
                {
                    "job_id": job["job_id"],
                    "worker_job_id": os.environ.get("SLURM_JOB_ID"),
                    "worker_pool": "remote",
                    "pid": os.getpid(),
                    "started": time.time(),
                }
            ).encode(),
        )
        committed = False
        target = local / job["job_id"]
        try:
            version = job["metadata"]["policy_version"]
            started = time.monotonic()
            await asyncio.to_thread(pin_job, directory, target)
            await asyncio.to_thread(validate_job, target, job)
            stage_seconds = time.monotonic() - started
            backend.assert_idle()
            started = time.monotonic()
            async with asyncio.timeout(study.weight_transfer_timeout_seconds):
                await update_weights(backend.orch.admin_clients.clients, target / "weights", step=version)
            load_seconds = time.monotonic() - started
            backend.orch.policy.version = version
            backend.orch.watcher.ckpt_step = version
            backend.orch.progress.step = version + 1
            await backend.orch.watcher._notify_update(version)
            grading = study.output_dir / "grading.jsonl"
            grading_offset = grading.stat().st_size if grading.exists() else 0
            started = time.monotonic()
            cohort = await backend.generate(version, "deferred")
            generation_seconds = time.monotonic() - started
            backend.assert_idle()
            archive = study.output_dir / "rollouts" / f"{version}-deferred-{cohort.digest}.msgpack"
            body = msgspec.msgpack.decode(archive.read_bytes())
            body["grading_records"] = grading_records(grading, grading_offset, cohort.response_ids)
            await asyncio.to_thread(
                publish_result,
                directory,
                job,
                msgspec.msgpack.encode(body),
                {
                    "output_tokens": cohort.output_tokens,
                    "generation_wall_seconds": generation_seconds,
                    "historical_stage_seconds": stage_seconds,
                    "historical_load_seconds": load_seconds,
                    "worker_job_id": os.environ.get("SLURM_JOB_ID"),
                },
            )
            committed = True
            print(
                json.dumps(
                    {"historical_version": version, "job_id": job["job_id"], "generation_seconds": generation_seconds}
                ),
                flush=True,
            )
            try:
                await asyncio.to_thread(shutil.rmtree, target)
            except OSError as error:
                print(json.dumps({"cleanup_warning": str(error), "path": str(target)}), flush=True)
            return True
        except BaseException as error:
            if not committed:
                atomic_write(
                    directory / "failure.json",
                    json.dumps(
                        {"job_id": job["job_id"], "error_type": type(error).__name__, "error": str(error)}
                    ).encode(),
                )
            raise


async def supervise_learner(work, learner_job, interval=5):
    async def watch():
        while await asyncio.to_thread(learner_alive, learner_job):
            await asyncio.sleep(interval)

    task = asyncio.create_task(work)
    watcher = asyncio.create_task(watch())
    try:
        done, _ = await asyncio.wait((task, watcher), return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            await task
        else:
            await watcher
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            print(json.dumps({"shutdown": "learner_allocation_ended", "learner_job_id": learner_job}), flush=True)
    finally:
        for pending in (task, watcher):
            if not pending.done():
                pending.cancel()
        await asyncio.gather(task, watcher, return_exceptions=True)


async def consume(study, config_path, learner_job):
    from prime_rl import monitors
    from prime_rl.configs.orchestrator import OrchestratorConfig
    from prime_rl.orchestrator import orchestrator as module

    from deepseek_study.rollouts.controller import PrimeBackend
    from deepseek_study.rollouts.history_store import load_result
    from deepseek_study.runtime.identity import capture, read_identity

    config = OrchestratorConfig.model_validate_json(config_path.read_text())
    if config.weight_broadcast.type != "filesystem" or config.resume is not None:
        raise ValueError("Historical workers require files-only policy loading without trainer resume")
    session = json.loads((study.historical_rollouts / "session.json").read_text())
    identity = read_identity(study.output_dir / "source" / "identity.json")
    validate_session(study, session, learner_job, identity)
    if capture(Path(__file__).resolve().parents[3], study) != identity:
        raise ValueError("Historical worker source changed after initialization")
    orch = module.Orchestrator(config)
    local = study.output_dir / "jobs"
    local.mkdir()

    async def run():
        await setup_manual_receiver(module, orch)
        backend = PrimeBackend(study, orch)
        shutdown = asyncio.Event()
        completed = set()
        expected = remote_versions(session["contract"])

        async def dispatch():
            await orch.dispatcher.start()
            if not shutdown.is_set():
                raise RuntimeError("Historical dispatcher stopped before worker shutdown")

        async with asyncio.TaskGroup() as tasks:
            dispatcher = tasks.create_task(dispatch())
            while True:
                entries = []
                for path in (study.historical_rollouts / "jobs").glob("*/job.json"):
                    job = json.loads(path.read_text())
                    version = validate_session_job(job, session, path.parent)
                    entries.append((version, path.parent, job))
                if len({entry[0] for entry in entries}) != len(entries):
                    raise ValueError("A learner session published multiple jobs for one policy version")
                progressed = False
                for version, directory, job in sorted(entries, key=lambda item: (item[0], item[2]["job_id"])):
                    if job["metadata"]["worker_pool"] != "remote" or version in completed:
                        continue
                    produced = await process_job(study, backend, directory, job, local)
                    progressed = produced or progressed
                    if produced or (
                        (directory / "result.json").exists()
                        and await asyncio.to_thread(load_result, directory, job) is not None
                    ):
                        completed.add(version)
                if completed == expected:
                    atomic_write(
                        study.output_dir / "historical-complete.json",
                        json.dumps(
                            {
                                "learner_job_id": learner_job,
                                "run_uuid": session["run_uuid"],
                                "completed_versions": sorted(completed),
                                "worker_pool": "remote",
                            }
                        ).encode(),
                    )
                    shutdown.set()
                    await orch.dispatcher.stop()
                    await dispatcher
                    return
                if not progressed:
                    await asyncio.sleep(2)

    try:
        await supervise_learner(run(), learner_job)
    finally:
        try:
            await stop_orchestrator(orch)
        finally:
            await monitors.finalize()


def launch(study, root, learner_job):
    import torch
    from prime_rl.entrypoints.rl import env_servers
    from prime_rl.utils.process import DEFAULT_COMMON_ENV_VARS, DEFAULT_INFERENCE_ENV_VARS

    from deepseek_study.dataset.assets import validate_prepared
    from deepseek_study.runtime.build import build
    from deepseek_study.runtime.identity import capture, snapshot
    from deepseek_study.runtime.launcher import inference_command, verify_upstream
    from deepseek_study.runtime.processes import stop_process_groups

    if study.historical_rollouts is None:
        raise ValueError("Historical workers require an explicit shared rollout root")
    verify_upstream(root)
    validate_prepared(study)
    if torch.cuda.device_count() != study.inference_gpus:
        raise ValueError("Historical worker must use every allocated GPU for inference")
    output = study.output_dir
    output.mkdir(parents=True, exist_ok=False)
    session_path = study.historical_rollouts / "session.json"
    started = time.monotonic()
    while not session_path.exists():
        if not learner_alive(learner_job):
            raise RuntimeError("Learner ended before historical worker initialization")
        if time.monotonic() - started > study.generation_timeout_seconds:
            raise TimeoutError("No historical learner session appeared")
        time.sleep(2)
    session = json.loads(session_path.read_text())
    identity = capture(root, study)
    validate_session(study, session, learner_job, identity)
    snapshot(root, output / "source", identity)
    config = build(study, output / "configs")
    processes = {}
    env = {
        **os.environ,
        **DEFAULT_COMMON_ENV_VARS,
        "DEEPSEEK_STUDY_RUNBOARD": "0",
        "PRL_RUN_ID": "historical-" + os.environ["SLURM_JOB_ID"],
        "PRL_RUN_NAME": output.name,
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
    }
    logs = output / "logs"
    logs.mkdir()

    def start(name, args, overrides):
        with (logs / (name + ".log")).open("x") as stream:
            processes[name] = subprocess.Popen(
                args, env={**env, **overrides}, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )

    def interrupted(*_):
        raise KeyboardInterrupt("Historical worker received termination")

    previous_handlers = {number: signal.signal(number, interrupted) for number in (signal.SIGTERM, signal.SIGINT)}
    try:
        start(
            "inference",
            inference_command(study, output / "configs/inference.json"),
            DEFAULT_INFERENCE_ENV_VARS,
        )
        for split, source, _ in env_servers(config):
            start(
                "env-" + source.resolved_name,
                [
                    sys.executable,
                    "-m",
                    "prime_rl.entrypoints.env_server",
                    "@",
                    str(output / "configs/envs" / split / (source.resolved_name + ".json")),
                ],
                {"CUDA_VISIBLE_DEVICES": ""},
            )
        start(
            "controller",
            [
                sys.executable,
                "-m",
                "deepseek_study.runtime.historical_worker",
                "consume",
                str(output / "configs/study.json"),
                "--learner-job",
                str(learner_job),
            ],
            {"CUDA_VISIBLE_DEVICES": ""},
        )
        while processes["controller"].poll() is None:
            if not learner_alive(learner_job):
                print(json.dumps({"shutdown": "learner_allocation_ended", "learner_job_id": learner_job}), flush=True)
                return
            if any(p.poll() is not None for name, p in processes.items() if name != "controller"):
                raise RuntimeError("Historical inference or environment exited")
            time.sleep(2)
        if processes["controller"].returncode:
            raise RuntimeError("Historical controller failed; inspect its log")
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
        errors = stop_process_groups(processes)
        if errors:
            print(json.dumps({"shutdown_errors": errors}), file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["launch", "consume"])
    parser.add_argument("study", type=Path)
    parser.add_argument("--learner-job", required=True, type=int)
    args = parser.parse_args()
    study = StudyConfig.read(args.study)
    if args.mode == "launch":
        launch(study, Path(__file__).resolve().parents[3], args.learner_job)
    else:
        asyncio.run(consume(study, args.study.parent / "orchestrator.json", args.learner_job))


if __name__ == "__main__":
    main()
