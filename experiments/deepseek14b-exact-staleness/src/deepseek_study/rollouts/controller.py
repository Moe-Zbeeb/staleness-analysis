import asyncio
import hashlib
import json
import math
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, replace
from pathlib import Path

import msgspec

from deepseek_study.runtime import checkpoints
from deepseek_study.runtime.identity import read_identity
from deepseek_study.learning.advantages import StudyGRPO
from deepseek_study.rollouts.queue import Cohort, QueueState, run
from deepseek_study.rollouts.provenance import ProvenanceTrainSink
from deepseek_study.dataset.rewards import completion_tokens
from prime_rl import monitors
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.types import DispatchFailure, GroupCancellation
from prime_rl.orchestrator.curriculum.samplers.standard import StandardSampler
from prime_rl.orchestrator.utils import train_work
from prime_rl.transports.batch import TrainingSample


@dataclass(frozen=True)
class Payload:
    samples: bytes
    policy_spans: tuple[tuple[int, int], ...]
    rewards: tuple[float, ...]
    truncated: tuple[bool, ...]
    task_keys: tuple[str, ...]
    sample_response_ids: tuple[str, ...]
    sample_task_keys: tuple[str, ...]


def payload_digest(payload):
    return hashlib.sha256(msgspec.msgpack.encode(payload)).hexdigest()


class FiniteSource:
    def __init__(self, source):
        self.source = source
        self.requests = deque()

    def sampler(self):
        if len(self.source.curricula) != 1:
            raise ValueError("Exact-staleness prompt scheduling requires one fixed dataset")
        curriculum = next(iter(self.source.curricula.values()))
        if curriculum.gates or not isinstance(curriculum.sampler, StandardSampler) or curriculum.sampler.tasks is None:
            raise ValueError(
                "Exact-staleness prompt scheduling requires a finite standard sampler without admission gates"
            )
        return curriculum.sampler

    def expected_questions(self, cohort_index, prompts, responses):
        tasks = self.sampler().tasks
        return Counter(
            str(tasks[(cohort_index * prompts + index) % len(tasks)].data.question_id)
            for index in range(prompts)
            for _ in range(responses)
        )

    def begin(self, version, prompts, responses, cohort_index=None):
        if self.requests:
            raise RuntimeError("Previous cohort still has undispatched prompts")
        if cohort_index is not None:
            sampler = self.sampler()
            cursor = cohort_index * prompts
            if sampler.cursor != cursor:
                sampler.load_state_dict({"cursor": cursor})
        for _ in range(prompts):
            request = self.source.next_task(step=version + 1)
            if request is None:
                raise RuntimeError("Task sampler stopped before the cohort was complete")
            self.requests.append(replace(request, step=version + 1))

    def next_task(self, *, step):
        return self.requests.popleft() if self.requests else None


class PrimeBackend:
    def __init__(self, study, orchestrator):
        self.study = study
        self.orch = orchestrator
        self.orch.train_sink = ProvenanceTrainSink(self.orch.train_sink)
        self.source = FiniteSource(orchestrator.train_source)
        self.orch.dispatcher.train_source = self.source
        self.orch.concurrency.bind(
            set_limit=self.orch.dispatcher.set_limit,
            get_inflight=lambda: self.orch.dispatcher.current_inflight,
        )
        for env in self.orch.train_envs:
            env.algorithm = StudyGRPO(env.config.algo, self.orch.clients, study)
        self.shipped = None
        self.step_started = time.monotonic()
        self.training_wait_seconds = 0.0
        self.weight_transfer_seconds = 0.0
        self.identity = read_identity(study.output_dir / "source" / "identity.json")

    def assert_idle(self):
        dispatcher = self.orch.dispatcher
        if dispatcher.groups or dispatcher.inflight or not dispatcher.out_q.empty() or self.source.requests:
            raise RuntimeError("Cannot change inference weights while a rollout cohort is active")
        sink = self.orch.train_sink
        if sink.buffered_count() or sink.pending_batch:
            raise RuntimeError("A partial training group remains in the upstream sink")

    async def generate(self, version, purpose):
        started = time.monotonic()
        async with asyncio.timeout(self.study.generation_timeout_seconds):
            self.assert_idle()
            if self.orch.policy.version != version:
                raise RuntimeError("Inference weights do not match the requested cohort version")
            cohort_index = version + self.study.lag if purpose == "deferred" else version
            self.source.begin(version, self.study.prompts_per_update, self.study.responses_per_prompt, cohort_index)
            episodes, batch = [], None
            for _ in range(self.study.response_batch_size):
                episode = await self.orch.dispatcher.out_q.get()
                if isinstance(episode, (DispatchFailure, GroupCancellation)):
                    raise RuntimeError(f"Cohort failed; refusing replacement sampling: {episode}")
                if not episode.ok or len(episode.traces) != 1 or episode.traces[0].has_error:
                    failure_id = hashlib.sha256(str(episode.id).encode()).hexdigest()
                    checkpoints.atomic_write(
                        self.study.output_dir / "failures" / f"{failure_id}.json", episode.model_dump_json().encode()
                    )
                    raise RuntimeError("Cohort contains an invalid or failed episode")
                span = train_work(episode).policy
                if span is None or span.start != version or span.end != version:
                    raise RuntimeError("A response spans more than one inference policy version")
                trace = episode.traces[0]
                tokens = completion_tokens(trace)
                if not tokens or len(tokens) > self.study.response_max_tokens:
                    raise RuntimeError("Response length differs from the study contract")
                episodes.append(episode)
                self.append(
                    "grading.jsonl",
                    {
                        "response_id": str(episode.id),
                        "policy_version": version,
                        "question_id": str(episode.task.data.question_id),
                        "grading": trace.info.get("study_grading"),
                    },
                )
                await monitors.log([episode], version + 1, "train", "all")
                ready = await self.orch.train_sink.add(episode)
                if ready is not None:
                    if batch is not None:
                        raise RuntimeError("One cohort produced multiple training batches")
                    batch = ready
            self.assert_idle()
            if batch is None or len(batch.samples) != self.study.response_batch_size or batch.failures:
                raise RuntimeError("Upstream sink changed the complete cohort")
            for sample in batch.samples:
                if len(sample.token_ids) > self.study.sequence_length:
                    raise RuntimeError("Packing would truncate a response")
                if any(not math.isfinite(value) for value in sample.logprobs):
                    raise RuntimeError("Nonfinite behavior log-probabilities")
            payload = Payload(
                samples=msgspec.msgpack.encode(batch.samples),
                policy_spans=tuple((train_work(e).policy.start, train_work(e).policy.end) for e in episodes),
                rewards=tuple(e.traces[0].reward for e in episodes),
                truncated=tuple(e.traces[0].is_truncated for e in episodes),
                task_keys=tuple(str(e.task.data.question_id) for e in episodes),
                sample_response_ids=tuple(item[0] for item in self.orch.train_sink.sample_provenance),
                sample_task_keys=tuple(item[1] for item in self.orch.train_sink.sample_provenance),
            )
            cohort = Cohort(
                version=version,
                response_ids=tuple(str(e.id) for e in episodes),
                output_tokens=sum(len(completion_tokens(e.traces[0])) for e in episodes),
                payload=payload,
                digest=payload_digest(payload),
            )
            await asyncio.to_thread(
                checkpoints.atomic_write,
                self.study.output_dir / "rollouts" / f"{version}-{purpose}-{cohort.digest}.msgpack",
                msgspec.msgpack.encode(
                    {
                        "format": 2,
                        "behavior_version": version,
                        "purpose": purpose,
                        "response_ids": cohort.response_ids,
                        "payload": payload,
                    }
                ),
            )
            self.append(
                "generations.jsonl",
                {
                    "policy_version": version,
                    "purpose": purpose,
                    "consumption_step": cohort_index + 1,
                    "response_ids": cohort.response_ids,
                    "task_keys": payload.task_keys,
                    "output_tokens": cohort.output_tokens,
                    "digest": cohort.digest,
                    "generation_wall_seconds": time.monotonic() - started,
                },
            )
            return cohort

    async def ship(self, cohort, learner_version):
        if self.orch.progress.step != learner_version + 1 or self.orch.policy.version != learner_version:
            raise RuntimeError("Learner, dispatcher, and queue clocks disagree")
        payload = cohort.payload
        if payload_digest(payload) != cohort.digest:
            raise RuntimeError("Stored tokens, log-probabilities, advantages, or provenance changed")
        if len(payload.policy_spans) != self.study.response_batch_size:
            raise RuntimeError("Response provenance is incomplete")
        questions = dict(zip(cohort.response_ids, payload.task_keys, strict=True))
        if (
            Counter(payload.sample_response_ids) != Counter(cohort.response_ids)
            or len(payload.sample_task_keys) != self.study.response_batch_size
            or any(
                questions[response] != question
                for response, question in zip(payload.sample_response_ids, payload.sample_task_keys, strict=True)
            )
        ):
            raise RuntimeError("Training sample provenance differs from the rollout cohort")
        if any(start != cohort.version or end != cohort.version for start, end in payload.policy_spans):
            raise RuntimeError("Mixed behavior versions in a queued cohort")
        expected = 0 if learner_version < self.study.lag else self.study.lag
        if learner_version - cohort.version != expected:
            raise RuntimeError("Incorrect optimizer-update age")
        if Counter(payload.task_keys) != self.source.expected_questions(
            learner_version, self.study.prompts_per_update, self.study.responses_per_prompt
        ):
            raise RuntimeError("Cohort questions do not match their assigned consumption update")
        samples = msgspec.msgpack.decode(payload.samples, type=list[TrainingSample])
        if len(samples) != self.study.response_batch_size:
            raise RuntimeError("Stored training sample count changed")
        grid = await asyncio.to_thread(self.orch.packer.pack, samples)
        await self.orch.sender.send(grid)
        self.orch.progress.step += 1
        self.orch.progress.total_samples += len(samples)
        self.orch.progress.total_problems += self.study.prompts_per_update
        self.orch.progress.total_tokens += cohort.output_tokens
        self.shipped = cohort

    async def synchronize(self, version):
        self.assert_idle()
        started = time.monotonic()
        try:
            await asyncio.wait_for(
                self.orch.watcher.receiver.wait_published(version), self.study.training_timeout_seconds
            )
        except TimeoutError as error:
            raise TimeoutError(f"Learner policy {version} was not published before the training deadline") from error
        self.training_wait_seconds = time.monotonic() - started
        started = time.monotonic()
        try:
            await asyncio.wait_for(
                self.orch.watcher.apply_policy_update(version), self.study.weight_transfer_timeout_seconds
            )
        except TimeoutError as error:
            raise TimeoutError(f"Policy {version} transfer exceeded the weight-transfer deadline") from error
        self.weight_transfer_seconds = time.monotonic() - started
        if self.orch.policy.version != version or self.orch.watcher.ckpt_step != version:
            raise RuntimeError("Inference did not acknowledge the completed learner update")

    def append(self, name, record):
        with (self.study.output_dir / name).open("a") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if name != "grading.jsonl":
                os.fsync(stream.fileno())

    async def commit(self, state, receipt):
        payload = self.shipped.payload
        receipt.update(
            {
                "mean_reward": sum(payload.rewards) / len(payload.rewards),
                "truncation_fraction": sum(payload.truncated) / len(payload.truncated),
                "response_ids": self.shipped.response_ids,
                "question_ids": payload.task_keys,
                "consumed_prompt_position": state.completed_steps * self.study.prompts_per_update,
                "generation_prompt_cursor": self.source.sampler().cursor,
                "zero_advantage_fraction": sum(
                    all(value == 0 for value in sample.advantages)
                    for sample in msgspec.msgpack.decode(payload.samples, type=list[TrainingSample])
                )
                / self.study.response_batch_size,
                "step_wall_seconds": time.monotonic() - self.step_started,
                "queue_payload_bytes": sum(len(cohort.payload.samples) for cohort in state.pending.values()),
                "training_wait_seconds": self.training_wait_seconds,
                "weight_transfer_seconds": self.weight_transfer_seconds,
            }
        )
        self.append("updates.jsonl", receipt)
        print(
            json.dumps({key: receipt[key] for key in ("step", "age_min", "age_max", "warmup", "mean_reward")}),
            flush=True,
        )
        if (
            state.completed_steps % self.study.checkpoint_interval == 0
            or state.completed_steps == self.study.max_steps
        ):
            directory = self.study.output_dir / "checkpoints" / f"step_{state.completed_steps}"
            async with asyncio.timeout(self.study.checkpoint_timeout_seconds):
                while not (directory / "trainer" / ".metadata").is_file() or any(
                    not (directory / "rng" / f"rank_{rank}.pt").is_file() for rank in range(self.study.trainer_gpus)
                ):
                    await asyncio.sleep(0.2)
                self.orch.ckpt_manager.save(self.orch.progress, self.orch.train_source, step=state.completed_steps)
                components_hash = await asyncio.to_thread(checkpoints.seal, directory, self.study.trainer_gpus)
                await asyncio.to_thread(
                    checkpoints.save,
                    directory / "study",
                    state,
                    self.study.fingerprint(),
                    self.identity["sha256"],
                    components_hash,
                )
                await asyncio.to_thread(
                    checkpoints.prune_complete,
                    directory.parent,
                    self.study.checkpoint_keep_last,
                    self.study.checkpoint_keep_interval,
                    self.identity["sha256"],
                )
        self.step_started = time.monotonic()


async def control(study, orchestrator_path, resume=None):
    from prime_rl.orchestrator.orchestrator import Orchestrator

    config = OrchestratorConfig.model_validate_json(Path(orchestrator_path).read_text())
    identity = read_identity(study.output_dir / "source" / "identity.json")
    if resume:
        checkpoints.verify_components(Path(resume))
    state = (
        checkpoints.load(Path(resume) / "study", study.fingerprint(), identity["sha256"])
        if resume
        else QueueState(study.lag)
    )
    orch = Orchestrator(config)
    await orch.setup()
    try:
        if orch.policy.version != state.completed_steps or orch.progress.step != state.completed_steps + 1:
            raise RuntimeError("Restored trainer/orchestrator state does not match the rollout queue")
        backend = PrimeBackend(study, orch)
        async with asyncio.TaskGroup() as tasks:
            dispatcher = tasks.create_task(orch.dispatcher.start())
            await run(backend, state, study.max_steps, study.response_batch_size)
            await orch.dispatcher.stop()
            await dispatcher
        await monitors.finalize()
        checkpoints.atomic_write(
            study.output_dir / "study-complete.json",
            json.dumps(
                {"step": state.completed_steps, "lag": study.lag, "identity_sha256": identity["sha256"]}
            ).encode(),
        )
    finally:
        await orch.stop()
