import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from deepseek_study.rollouts.queue import Cohort, QueueState


@dataclass
class AsyncQueueState(QueueState):
    jobs: dict[int, dict[str, Any]] = field(default_factory=dict)

    def validate(self):
        if self.completed_steps < 0 or (self.generation_stop is not None and self.generation_stop < 0):
            raise ValueError("Invalid queue progress")
        end = self.completed_steps if self.generation_stop is None else min(self.completed_steps, self.generation_stop)
        expected = set(range(max(0, self.completed_steps - self.lag), end))
        if self.lag < 0 or set(self.pending) & set(self.jobs) or set(self.pending) | set(self.jobs) != expected:
            raise ValueError("Asynchronous queue does not match the completed policy version")
        for job in self.jobs.values():
            validate_job(job)
        for version, cohort in self.pending.items():
            if (
                cohort.version != version
                or len(set(cohort.response_ids)) != len(cohort.response_ids)
                or not set(cohort.response_ids).issubset(self.generated_ids)
            ):
                raise ValueError("Queue checkpoint has invalid response provenance")
        pending_ids = [response for cohort in self.pending.values() for response in cohort.response_ids]
        if len(pending_ids) != len(set(pending_ids)):
            raise ValueError("Pending cohorts contain duplicated responses")
        if self.generated_cohorts != self.completed_steps + len(self.pending):
            raise ValueError("Generated, consumed, and completed cohort accounting does not balance")


def validate_job(job):
    if not isinstance(job, dict) or not job:
        raise ValueError("A deferred job must be a nonempty serializable dictionary")
    try:
        json.dumps(job, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("A deferred job must be a serializable dictionary") from error


class AsyncBackend(Protocol):
    async def generate(self, version: int, purpose: str) -> Cohort: ...
    async def submit_deferred(self, version: int) -> dict[str, Any]: ...
    async def poll_deferred(self, job: dict[str, Any]) -> Cohort | None: ...
    async def wait_deferred(self, job: dict[str, Any]) -> Cohort: ...
    async def ship(self, cohort: Cohort, learner_version: int) -> None: ...
    async def synchronize(self, version: int) -> None: ...
    async def commit(self, state: AsyncQueueState, receipt: dict) -> None: ...


async def run_async(backend: AsyncBackend, state: AsyncQueueState, max_steps: int, responses: int):
    if not isinstance(state, AsyncQueueState):
        raise TypeError("Asynchronous generation requires an asynchronous queue checkpoint")
    if max_steps <= state.lag or state.completed_steps > max_steps or responses < 1:
        raise ValueError("Invalid exact-staleness run horizon or cohort size")
    if state.generation_stop is None:
        state.generation_stop = max(0, max_steps - state.lag)
    if state.generation_stop != max(0, max_steps - state.lag):
        raise ValueError("The queue checkpoint belongs to another training horizon")
    state.validate()

    def materialize(version, cohort):
        if not isinstance(cohort, Cohort):
            raise TypeError("Deferred generation did not produce a complete cohort")
        state.record_generation(cohort, version, responses)
        state.pending[version] = cohort
        del state.jobs[version]

    async def poll_ready():
        for version in sorted(state.jobs):
            cohort = await backend.poll_deferred(state.jobs[version])
            if cohort is not None:
                materialize(version, cohort)

    while state.completed_steps < max_steps:
        version = state.completed_steps
        warmup = version < state.lag
        if state.lag and version < state.generation_stop:
            job = await backend.submit_deferred(version)
            validate_job(job)
            state.jobs[version] = json.loads(json.dumps(job, allow_nan=False))
        if warmup or state.lag == 0:
            training = await backend.generate(version, "warmup" if warmup else "on_policy")
            state.record_generation(training, version, responses)
        else:
            required = version - state.lag
            if required in state.jobs:
                materialize(required, await backend.wait_deferred(state.jobs[required]))
            training = state.pending.pop(required)
        expected_age = 0 if warmup else state.lag
        if version - training.version != expected_age:
            raise ValueError("Refusing an update with the wrong response age")
        await backend.ship(training, version)
        await backend.synchronize(version + 1)
        state.completed_steps += 1
        await poll_ready()
        state.validate()
        await backend.commit(
            state,
            {
                "step": state.completed_steps,
                "learner_version": version,
                "behavior_version": training.version,
                "age_min": expected_age,
                "age_max": expected_age,
                "warmup": warmup,
                "responses": len(training.response_ids),
                "queued_versions": sorted(set(state.pending) | set(state.jobs)),
                "ready_versions": sorted(state.pending),
                "deferred_job_versions": sorted(state.jobs),
                "generated_cohorts": state.generated_cohorts,
                "generated_responses": len(state.generated_ids),
                "generated_output_tokens": state.generated_output_tokens,
                "payload_digest": training.digest,
            },
        )
