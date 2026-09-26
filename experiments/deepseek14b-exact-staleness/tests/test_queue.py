import asyncio

import pytest

from deepseek_study.runtime import checkpoints
from deepseek_study.rollouts.queue import Cohort, QueueState, run


class Backend:
    def __init__(self, version=0, stop_at=None):
        self.version = version
        self.receipts = []
        self.generated = []
        self.shipped = []
        self.active_generation = False
        self.overlapped = False
        self.stop_at = stop_at
        self.directory = None

    async def generate(self, version, purpose):
        assert self.version == version
        self.active_generation = True
        await asyncio.sleep(0)
        assert self.version == version
        self.active_generation = False
        ids = tuple(f"{version}-{purpose}-{i}" for i in range(4))
        cohort = Cohort(version, ids, 12, b"original_logprobs", "digest")
        self.generated.append(cohort)
        return cohort

    async def ship(self, cohort, learner_version):
        assert self.version == learner_version
        self.overlapped |= self.active_generation
        self.shipped.append((learner_version, cohort))

    async def synchronize(self, version):
        assert not self.active_generation
        assert version == self.version + 1
        self.version = version

    async def commit(self, state, receipt):
        self.receipts.append(receipt)
        if self.directory:
            checkpoints.save(self.directory, state, "protocol", "source")
        if state.completed_steps == self.stop_at:
            raise RuntimeError("simulated interruption")


@pytest.mark.parametrize("budget", [33, 40, 64, 97])
async def test_exact_32_and_no_unused_tail_rollouts(budget):
    backend, state = Backend(), QueueState(32)
    await run(backend, state, budget, 4)
    assert [receipt["age_min"] for receipt in backend.receipts] == [0] * 32 + [32] * (budget - 32)
    assert state.generated_cohorts == budget
    assert not state.pending
    assert len({item for _, cohort in backend.shipped for item in cohort.response_ids}) == budget * 4
    assert backend.overlapped


async def test_resume_restores_unconsumed_responses_without_regeneration(tmp_path):
    first, state = Backend(stop_at=35), QueueState(32)
    first.directory = tmp_path
    with pytest.raises(RuntimeError, match="interruption"):
        await run(first, state, 70, 4)
    restored = checkpoints.load(tmp_path, "protocol", "source")
    queued_ids = {key: cohort.response_ids for key, cohort in restored.pending.items()}
    second = Backend(version=35)
    await run(second, restored, 70, 4)
    for version, cohort in second.shipped:
        if version - 32 in queued_ids:
            assert cohort.response_ids == queued_ids[version - 32]
    assert restored.generated_cohorts == 70


async def test_corrupt_checkpoint_and_changed_protocol_are_rejected(tmp_path):
    backend = Backend(stop_at=2)
    backend.directory = tmp_path
    with pytest.raises(RuntimeError):
        await run(backend, QueueState(32), 40, 4)
    with pytest.raises(ValueError, match="protocol"):
        checkpoints.load(tmp_path, "other", "source")
    with (tmp_path / "queue.pkl").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        checkpoints.load(tmp_path, "protocol", "source")


async def test_failed_generation_never_advances_policy():
    backend = Backend()

    async def fail(version, purpose):
        if purpose == "deferred":
            raise RuntimeError("generation failed")
        return Cohort(version, tuple(str(i) for i in range(4)), 4, b"", "")

    backend.generate = fail
    with pytest.raises(ExceptionGroup):
        await run(backend, QueueState(32), 40, 4)
    assert backend.version == 0
    assert not backend.receipts


@pytest.mark.parametrize("lag", [0, 1, 2, 8, 32, 64])
@pytest.mark.parametrize("extra", [1, 5, 70])
async def test_requested_lag_is_exact_for_every_measured_update(lag, extra):
    backend, state = Backend(), QueueState(lag)
    await run(backend, state, lag + extra, 4)
    assert [row["age_min"] for row in backend.receipts] == [0] * lag + [lag] * extra
    assert all(row["age_min"] == row["age_max"] for row in backend.receipts)
    assert state.generated_cohorts == lag + extra
    assert not state.pending


@pytest.mark.parametrize("lag,stop", [(0, 3), (1, 1), (8, 3), (8, 8), (8, 11), (8, 16)])
async def test_resume_during_bootstrap_steady_state_and_queue_drain(lag, stop, tmp_path):
    budget = 19
    backend = Backend(stop_at=stop)
    backend.directory = tmp_path
    with pytest.raises(RuntimeError, match="interruption"):
        await run(backend, QueueState(lag), budget, 4)
    state = checkpoints.load(tmp_path, "protocol", "source")
    following = Backend(version=stop)
    await run(following, state, budget, 4)
    used = backend.shipped + following.shipped
    assert len(used) == budget
    assert len({response for _, cohort in used for response in cohort.response_ids}) == 4 * budget
    assert all(version - cohort.version == (0 if version < lag else lag) for version, cohort in used)
    assert not state.pending


async def test_resume_rejects_changed_source_identity(tmp_path):
    backend = Backend(stop_at=1)
    backend.directory = tmp_path
    with pytest.raises(RuntimeError):
        await run(backend, QueueState(1), 3, 4)
    with pytest.raises(ValueError, match="protocol"):
        checkpoints.load(tmp_path, "protocol", "changed-source")
