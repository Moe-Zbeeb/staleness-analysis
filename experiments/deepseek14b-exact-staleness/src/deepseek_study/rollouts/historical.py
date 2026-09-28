import asyncio
import fcntl
import json
import math
import os
import shutil
import time
from collections import Counter

import msgspec
from prime_rl.transports.batch import TrainingSample

from deepseek_study import MODEL_ID, MODEL_REVISION
from deepseek_study.rollouts.controller import Payload, PrimeBackend, payload_digest
from deepseek_study.rollouts.history_store import (
    freeze_export,
    load_result,
    pin_job,
    publish_job,
    publish_result,
    validate_job,
)
from deepseek_study.rollouts.queue import Cohort
from deepseek_study.runtime.checkpoints import atomic_write, file_digest

CONTRACT_FIELDS = (
    "prompts_per_update",
    "responses_per_prompt",
    "prompt_max_tokens",
    "response_max_tokens",
    "max_steps",
    "temperature",
    "prompt_instruction",
    "truncated_reward",
    "reasoning_required",
    "seed",
    "lag",
    "advantage_normalization",
    "advantage_epsilon",
    "reward_timeout_seconds",
    "reward_outer_timeout_seconds",
    "reward_retries",
    "loss_reduction",
    "clip_epsilon",
)


def contract(study):
    values = {key: getattr(study, key) for key in CONTRACT_FIELDS}
    values.update(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        dataset_sha256=file_digest(study.dataset_path),
        manifest_sha256=file_digest(study.data_manifest),
    )
    return values


def worker_pool(version, lag):
    return "remote" if version < lag else "local"


def decode_cohort(data, job):
    metadata = job["metadata"]
    version = metadata["policy_version"]
    settings = metadata["contract"]
    if (
        type(version) is not int
        or not 0 <= version < settings["max_steps"] - settings["lag"]
        or metadata["consumption_step"] != version + settings["lag"] + 1
    ):
        raise ValueError("Historical job has an invalid policy or consumption version")
    archive = msgspec.msgpack.decode(data)
    if (
        not isinstance(archive, dict)
        or archive.get("format") != 2
        or archive.get("behavior_version") != version
        or archive.get("purpose") != "deferred"
    ):
        raise ValueError("Historical archive has the wrong policy version or purpose")
    payload = msgspec.convert(archive["payload"], type=Payload)
    response_ids = msgspec.convert(archive["response_ids"], type=tuple[str, ...])
    responses = settings["prompts_per_update"] * settings["responses_per_prompt"]
    arrays = (
        response_ids,
        payload.policy_spans,
        payload.rewards,
        payload.truncated,
        payload.task_keys,
        payload.sample_response_ids,
        payload.sample_task_keys,
    )
    if any(len(values) != responses for values in arrays) or len(set(response_ids)) != responses:
        raise ValueError("Historical cohort has incomplete or duplicate response provenance")
    if not 0 <= payload.verification_timeouts <= responses or any(
        not math.isfinite(value) for value in payload.rewards
    ):
        raise ValueError("Historical cohort has invalid grading metadata")
    if any(start != version or end != version for start, end in payload.policy_spans):
        raise ValueError("Historical response spans more than one policy version")
    questions = dict(zip(response_ids, payload.task_keys, strict=True))
    if Counter(payload.sample_response_ids) != Counter(response_ids) or any(
        questions[response] != question
        for response, question in zip(payload.sample_response_ids, payload.sample_task_keys, strict=True)
    ):
        raise ValueError("Historical training samples have incorrect response/question mappings")
    samples = msgspec.msgpack.decode(payload.samples, type=list[TrainingSample])
    if len(samples) != responses:
        raise ValueError("Historical cohort has an incomplete training sample group")
    output_tokens = 0
    for sample in samples:
        length = len(sample.token_ids)
        streams = (sample.mask, sample.logprobs, sample.temperatures, sample.advantages)
        if not length or any(values is None or len(values) != length for values in streams):
            raise ValueError("Historical training token streams are incomplete or misaligned")
        completion_tokens = sum(sample.mask)
        if (
            not 0 < completion_tokens <= settings["response_max_tokens"]
            or length > settings["prompt_max_tokens"] + settings["response_max_tokens"]
            or any(token < 0 for token in sample.token_ids)
        ):
            raise ValueError("Historical response length or token IDs differ from the study contract")
        if any(not math.isfinite(value) for values in (sample.logprobs, sample.advantages) for value in values):
            raise ValueError("Historical log-probabilities or advantages are nonfinite")
        if any(value != settings["temperature"] for value in sample.temperatures):
            raise ValueError("Historical sampling temperature differs from the study contract")
        if sample.ce_weights is not None or sample.ref_kl_weights is not None:
            raise ValueError("Historical samples contain an unexpected loss component")
        if sample.rl_weights is not None and (
            len(sample.rl_weights) != length or any(value != 1.0 for value in sample.rl_weights)
        ):
            raise ValueError("Historical samples change the GRPO token weighting")
        output_tokens += completion_tokens
    return Cohort(version, response_ids, output_tokens, payload, payload_digest(payload))


class HistoricalBackend(PrimeBackend):
    def __init__(self, study, orchestrator):
        super().__init__(study, orchestrator)
        self.root = study.historical_rollouts
        self.local = study.output_dir / "historical-exports"
        self.local.mkdir(exist_ok=True)
        self.publications = {}
        self.local_generations = {}
        self.accepted = set()
        self.registered_jobs = {}
        run = json.loads((study.output_dir / "run.json").read_text())
        self.run_uuid = run["run_uuid"]
        self.contract = contract(study)
        self.root.mkdir(parents=True, exist_ok=True)
        session = {
            "contract": self.contract,
            "identity": self.identity,
            "run_uuid": self.run_uuid,
            "config_sha256": study.fingerprint(),
            "learner_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        path = self.root / "session.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if run.get("resume_from"):
                if any(previous.get(key) != session[key] for key in ("contract", "identity", "config_sha256")):
                    raise ValueError("Historical recovery differs from the original study")
                self.run_uuid = previous["run_uuid"]
                session["run_uuid"] = self.run_uuid
            elif previous != session:
                raise ValueError("Historical rollout root belongs to a different study")
        atomic_write(path, json.dumps(session, sort_keys=True).encode())
        self.export_seconds = 0.0
        self.historical_wait_seconds = 0.0

    def register_job(self, version, job):
        metadata = job["metadata"]
        if (
            type(version) is not int
            or not 0 <= version < self.study.max_steps - self.study.lag
            or metadata.get("policy_version") != version
            or metadata.get("consumption_step") != version + self.study.lag + 1
            or metadata.get("contract") != self.contract
            or metadata.get("config_sha256") != self.study.fingerprint()
            or metadata.get("identity_sha256") != self.identity["sha256"]
            or metadata.get("worker_pool") != worker_pool(version, self.study.lag)
        ):
            raise ValueError("Historical job differs from the learner's scientific identity or version mapping")
        previous = self.registered_jobs.get(version)
        if previous is not None and previous != job["job_id"]:
            raise ValueError("Historical policy version is already bound to a different immutable export")
        self.registered_jobs[version] = job["job_id"]

    async def submit_deferred(self, version):
        started = time.monotonic()
        source = self.study.output_dir / "broadcasts" / f"step_{version}"
        if not (source / ".finished").is_file():
            raise ValueError("Cannot publish an incomplete policy export")
        metadata = {
            "run_uuid": self.run_uuid,
            "config_sha256": self.study.fingerprint(),
            "identity_sha256": self.identity["sha256"],
            "policy_version": version,
            "consumption_step": version + self.study.lag + 1,
            "contract": self.contract,
            "worker_pool": worker_pool(version, self.study.lag),
        }
        directory = self.local / f"step_{version}"
        job = await asyncio.to_thread(freeze_export, source, directory, metadata)
        self.register_job(version, job)
        if metadata["worker_pool"] == "local":
            self.local_generations[version] = asyncio.create_task(self.generate_local(job, directory))
        else:
            self.publications[job["job_id"]] = asyncio.create_task(
                asyncio.to_thread(publish_job, directory, self.root)
            )
        self.export_seconds = time.monotonic() - started
        return job

    async def generate_local(self, job, directory):
        version = job["metadata"]["policy_version"]
        with (directory / "worker.lock").open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("The local historical cohort is already claimed") from error
            if (directory / "attempt.json").exists():
                raise RuntimeError("An incomplete local historical attempt requires explicit recovery")
            atomic_write(
                directory / "attempt.json",
                json.dumps(
                    {
                        "job_id": job["job_id"],
                        "worker_job_id": os.environ.get("SLURM_JOB_ID"),
                        "worker_pool": "local",
                        "pid": os.getpid(),
                        "started": time.time(),
                    }
                ).encode(),
            )
            try:
                started = time.monotonic()
                cohort = await self.generate(version, "deferred")
                generation_seconds = time.monotonic() - started
                archive = self.study.output_dir / "rollouts" / f"{version}-deferred-{cohort.digest}.msgpack"
                data = archive.read_bytes()
                decoded = decode_cohort(data, job)
                if decoded != cohort:
                    raise ValueError("Local historical archive differs from the completed cohort")
                await asyncio.to_thread(
                    publish_result,
                    directory,
                    job,
                    data,
                    {
                        "output_tokens": cohort.output_tokens,
                        "generation_wall_seconds": generation_seconds,
                        "worker_job_id": os.environ.get("SLURM_JOB_ID"),
                        "worker_pool": "local",
                    },
                )
                self.accepted.add(job["job_id"])
                return cohort
            except BaseException as error:
                failure = json.dumps(
                    {"job_id": job["job_id"], "error_type": type(error).__name__, "error": str(error)}
                ).encode()
                atomic_write(directory / "failure.json", failure)
                raise

    async def synchronize(self, version):
        task = self.local_generations.get(version - 1)
        if task is not None:
            await task
            del self.local_generations[version - 1]
        await super().synchronize(version)

    async def restore_jobs(self, state, checkpoint):
        if checkpoint is None:
            if state.jobs:
                raise ValueError("Historical recovery requires pinned exports from its checkpoint")
            return
        for version, job in sorted(state.jobs.items()):
            self.register_job(version, job)
            if job["metadata"]["worker_pool"] == "local":
                raise RuntimeError("A committed checkpoint cannot contain an unfinished local historical cohort")
            source = checkpoint / "historical" / job["job_id"]
            target = self.local / f"step_{version}"
            await asyncio.to_thread(validate_job, source, job)
            await asyncio.to_thread(pin_job, source, target)
            shared = self.root / "jobs" / job["job_id"]
            existing = await asyncio.to_thread(load_result, shared, job) if (shared / "job.json").exists() else None
            if existing is not None:
                decode_cohort(existing, job)
                continue
            self.publications[job["job_id"]] = asyncio.create_task(asyncio.to_thread(publish_job, target, self.root))

    async def poll_deferred(self, job):
        version = job["metadata"]["policy_version"]
        if self.registered_jobs.get(version) != job["job_id"]:
            raise ValueError("Historical result does not belong to a submitted or restored job")
        self.register_job(version, job)
        generation = self.local_generations.get(version)
        if generation is not None:
            if not generation.done():
                return None
            await generation
        task = self.publications.get(job["job_id"])
        if task is not None:
            if not task.done():
                return None
            await task
        directory = (
            self.local / f"step_{version}"
            if job["metadata"]["worker_pool"] == "local"
            else self.root / "jobs" / job["job_id"]
        )
        data = await asyncio.to_thread(load_result, directory, job)
        if data is None:
            return None
        cohort = decode_cohort(data, job)
        receipt = json.loads((directory / "result.json").read_text())
        metrics = receipt["metrics"]
        if type(metrics.get("output_tokens")) is not int or metrics["output_tokens"] != cohort.output_tokens:
            raise ValueError("Historical result token count differs from the archived training samples")
        generation_seconds = metrics.get("generation_wall_seconds")
        if (
            isinstance(generation_seconds, bool)
            or not isinstance(generation_seconds, (int, float))
            or not math.isfinite(generation_seconds)
            or generation_seconds < 0
        ):
            raise ValueError("Historical result has an invalid generation duration")
        if Counter(cohort.payload.task_keys) != self.source.expected_questions(
            cohort.version + self.study.lag, self.study.prompts_per_update, self.study.responses_per_prompt
        ):
            raise ValueError("Historical cohort contains the wrong questions")
        if job["job_id"] not in self.accepted:
            atomic_write(
                self.study.output_dir / "rollouts" / f"{cohort.version}-deferred-{cohort.digest}.msgpack", data
            )
            self.append(
                "generations.jsonl",
                {
                    "policy_version": cohort.version,
                    "purpose": "deferred",
                    "consumption_step": job["metadata"]["consumption_step"],
                    "response_ids": cohort.response_ids,
                    "task_keys": cohort.payload.task_keys,
                    "digest": cohort.digest,
                    "output_tokens": cohort.output_tokens,
                    "generation_wall_seconds": generation_seconds,
                    "historical_job_id": job["job_id"],
                },
            )
            self.accepted.add(job["job_id"])
        return cohort

    async def wait_deferred(self, job):
        started = time.monotonic()
        async with asyncio.timeout(self.study.generation_timeout_seconds):
            while True:
                cohort = await self.poll_deferred(job)
                if cohort is not None:
                    self.historical_wait_seconds = time.monotonic() - started
                    return cohort
                await asyncio.sleep(1)

    async def commit(self, state, receipt):
        if any(job["metadata"]["worker_pool"] == "local" for job in state.jobs.values()):
            raise RuntimeError("A local historical cohort must be materialized before committing its update")
        receipt.update(
            historical_export_seconds=self.export_seconds,
            historical_wait_seconds=self.historical_wait_seconds,
            historical_pending_jobs=len(state.jobs),
        )
        if self.study.checkpoint_due(state.completed_steps) and state.jobs:
            destination = self.study.output_dir / "checkpoints" / f"step_{state.completed_steps}" / "historical"
            for version, job in sorted(state.jobs.items()):
                source = self.local / f"step_{version}"
                await asyncio.to_thread(validate_job, source, job)
                await asyncio.to_thread(pin_job, source, destination / job["job_id"])
        await super().commit(state, receipt)
        for directory in list(self.local.glob("step_*")):
            path = directory / "job.json"
            if path.exists():
                job = json.loads(path.read_text())
                if job["job_id"] in self.accepted:
                    await asyncio.to_thread(shutil.rmtree, directory)
                    if job["metadata"]["worker_pool"] == "remote":
                        weights = self.root / "jobs" / job["job_id"] / "weights"
                        if weights.exists():
                            await asyncio.to_thread(shutil.rmtree, weights)
        self.export_seconds = 0.0
        self.historical_wait_seconds = 0.0

    async def close(self):
        for task in self.local_generations.values():
            if not task.done():
                task.cancel()
        if self.local_generations:
            await asyncio.gather(*self.local_generations.values(), return_exceptions=True)
        if self.publications:
            await asyncio.gather(*self.publications.values(), return_exceptions=True)
