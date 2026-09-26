import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Cohort:
    version: int
    response_ids: tuple[str, ...]
    output_tokens: int
    payload: Any
    digest: str


@dataclass
class QueueState:
    lag: int
    generation_stop: int | None = None
    completed_steps: int = 0
    pending: dict[int, Cohort] = field(default_factory=dict)
    generated_ids: set[str] = field(default_factory=set)
    generated_cohorts: int = 0
    generated_output_tokens: int = 0

    def validate(self):
        if self.completed_steps < 0 or (self.generation_stop is not None and self.generation_stop < 0):
            raise ValueError("Invalid queue progress")
        end = self.completed_steps if self.generation_stop is None else min(self.completed_steps, self.generation_stop)
        expected = set(range(max(0, self.completed_steps - self.lag), end))
        if self.lag < 0 or set(self.pending) != expected:
            raise ValueError("Queue checkpoint does not match the completed policy version")
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
            raise ValueError("Generated, consumed, and queued cohort accounting does not balance")

    def record_generation(self, cohort, expected_version, expected_responses):
        ids = set(cohort.response_ids)
        if cohort.version != expected_version or len(cohort.response_ids) != expected_responses:
            raise ValueError("Cohort policy version or response count differs from the study")
        if len(ids) != expected_responses or ids & self.generated_ids:
            raise ValueError("A response was duplicated or reused")
        self.generated_ids.update(ids)
        self.generated_cohorts += 1
        self.generated_output_tokens += cohort.output_tokens


class Backend(Protocol):
    async def generate(self, version: int, purpose: str) -> Cohort: ...
    async def ship(self, cohort: Cohort, learner_version: int) -> None: ...
    async def synchronize(self, version: int) -> None: ...
    async def commit(self, state: QueueState, receipt: dict) -> None: ...


async def run(backend: Backend, state: QueueState, max_steps: int, responses: int):
    if max_steps <= state.lag or state.completed_steps > max_steps or responses < 1:
        raise ValueError("Invalid exact-staleness run horizon or cohort size")
    if state.generation_stop is None:
        state.generation_stop = max(0, max_steps - state.lag)
    if state.generation_stop != max(0, max_steps - state.lag):
        raise ValueError("The queue checkpoint belongs to another training horizon")
    state.validate()

    async def generate(version, purpose):
        cohort = await backend.generate(version, purpose)
        state.record_generation(cohort, version, responses)
        return cohort

    while state.completed_steps < max_steps:
        version = state.completed_steps
        warmup = version < state.lag
        if warmup or state.lag == 0:
            training = await generate(version, "warmup" if warmup else "on_policy")
        else:
            training = state.pending.pop(version - state.lag)
        expected_age = 0 if warmup else state.lag
        if version - training.version != expected_age:
            raise ValueError("Refusing an update with the wrong response age")
        if state.lag and version < state.generation_stop:
            async with asyncio.TaskGroup() as tasks:
                generation = tasks.create_task(generate(version, "deferred"))
                tasks.create_task(backend.ship(training, version))
            state.pending[version] = generation.result()
        else:
            await backend.ship(training, version)
        await backend.synchronize(version + 1)
        state.completed_steps += 1
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
                "queued_versions": sorted(state.pending),
                "generated_cohorts": state.generated_cohorts,
                "generated_responses": len(state.generated_ids),
                "generated_output_tokens": state.generated_output_tokens,
                "payload_digest": training.digest,
            },
        )
