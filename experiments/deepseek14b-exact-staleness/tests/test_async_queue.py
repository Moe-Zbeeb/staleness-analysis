import copy

import pytest

from deepseek_study.rollouts.async_queue import AsyncQueueState, run_async
from deepseek_study.rollouts.queue import Cohort, QueueState
from deepseek_study.runtime import checkpoints


def make_cohort(version, purpose="deferred"):
    return Cohort(version, tuple(f"{purpose}-{version}-{index}" for index in range(4)), 12, b"tokens", "digest")


class Backend:
    def __init__(self, version=0, stop_at=None, ready=None):
        self.version = version
        self.stop_at = stop_at
        self.ready = ready or {}
        self.submitted = []
        self.waited = []
        self.polled = []
        self.shipped = []
        self.receipts = []
        self.snapshots = []
        self.events = []
        self.directory = None

    async def submit_deferred(self, version):
        assert version == self.version
        self.events.append(("submit", version))
        self.submitted.append(version)
        return {"version": version, "export": {"sha256": f"immutable-policy-{version}"}}

    async def poll_deferred(self, job):
        version = job["version"]
        self.polled.append(version)
        if self.version >= self.ready.get(version, float("inf")):
            return make_cohort(version)
        return None

    async def wait_deferred(self, job):
        version = job["version"]
        self.events.append(("wait", version))
        self.waited.append((self.version, version))
        return make_cohort(version)

    async def generate(self, version, purpose):
        assert version == self.version
        self.events.append(("generate", version))
        return make_cohort(version, purpose)

    async def ship(self, cohort, learner_version):
        assert learner_version == self.version
        self.events.append(("ship", learner_version))
        self.shipped.append((learner_version, cohort))

    async def synchronize(self, version):
        assert version == self.version + 1
        self.version = version

    async def commit(self, state, receipt):
        state.validate()
        self.receipts.append(receipt)
        self.snapshots.append(copy.deepcopy(state))
        if self.directory:
            checkpoints.save(self.directory, state, "config", "identity")
        if state.completed_steps == self.stop_at:
            raise RuntimeError("interrupted")


@pytest.mark.parametrize("lag", [0, 1, 2, 8, 32, 256])
@pytest.mark.parametrize("extra", [1, 5, 70])
async def test_async_exact_age_and_no_unused_jobs(lag, extra):
    backend, state = Backend(), AsyncQueueState(lag)
    budget = lag + extra
    await run_async(backend, state, budget, 4)
    assert [receipt["age_min"] for receipt in backend.receipts] == [0] * lag + [lag] * extra
    assert all(receipt["age_min"] == receipt["age_max"] for receipt in backend.receipts)
    assert state.completed_steps == state.generated_cohorts == budget
    assert len(state.generated_ids) == budget * 4
    assert not state.pending and not state.jobs
    assert backend.submitted == (list(range(extra)) if lag else [])
    assert backend.waited == ([(version + lag, version) for version in range(extra)] if lag else [])


async def test_deferred_generation_does_not_block_bootstrap():
    backend = Backend(stop_at=3)
    state = AsyncQueueState(8)
    with pytest.raises(RuntimeError, match="interrupted"):
        await run_async(backend, state, 19, 4)
    assert state.completed_steps == 3
    assert set(state.jobs) == {0, 1, 2}
    assert not state.pending and not backend.waited
    for version in range(3):
        assert backend.events.index(("submit", version)) < backend.events.index(("generate", version))
        assert backend.events.index(("submit", version)) < backend.events.index(("ship", version))


async def test_out_of_order_results_are_materialized_and_consumed_by_version():
    backend = Backend(ready={0: 4, 1: 2, 2: 4, 3: 4})
    state = AsyncQueueState(4)
    await run_async(backend, state, 9, 4)
    assert backend.receipts[1]["ready_versions"] == [1]
    assert backend.receipts[1]["deferred_job_versions"] == [0]
    assert backend.receipts[3]["ready_versions"] == [0, 1, 2, 3]
    assert [cohort.version for _, cohort in backend.shipped] == [0, 1, 2, 3, 0, 1, 2, 3, 4]
    assert backend.waited == [(8, 4)]
    assert backend.polled[:3] == [0, 0, 1]


@pytest.mark.parametrize("lag,stop", [(0, 3), (1, 1), (8, 3), (8, 8), (8, 11), (8, 16)])
async def test_resume_preserves_ready_and_unfinished_cohorts(lag, stop, tmp_path):
    first = Backend(stop_at=stop, ready={version: version + 1 for version in range(0, 19, 2)})
    first.directory = tmp_path
    state = AsyncQueueState(lag)
    with pytest.raises(RuntimeError, match="interrupted"):
        await run_async(first, state, 19, 4)
    restored = checkpoints.load(tmp_path, "config", "identity")
    assert isinstance(restored, AsyncQueueState)
    pending_ids = {version: cohort.response_ids for version, cohort in restored.pending.items()}
    jobs = copy.deepcopy(restored.jobs)
    second = Backend(version=stop)
    await run_async(second, restored, 19, 4)
    used = first.shipped + second.shipped
    assert len(used) == 19
    assert len({response for _, cohort in used for response in cohort.response_ids}) == 76
    assert all(version - cohort.version == (0 if version < lag else lag) for version, cohort in used)
    for version, cohort in second.shipped:
        if version >= lag and cohort.version in pending_ids:
            assert cohort.response_ids == pending_ids[cohort.version]
    assert not set(jobs) & set(second.submitted)
    assert not restored.jobs and not restored.pending
    assert restored.generated_cohorts == 19


@pytest.mark.parametrize("via_poll", [False, True])
async def test_wrong_behavior_version_fails_closed(via_poll):
    backend = Backend()

    async def wrong(job):
        return make_cohort(job["version"] + 1)

    if via_poll:
        backend.poll_deferred = wrong
    else:
        backend.wait_deferred = wrong
    with pytest.raises(ValueError, match="policy version"):
        await run_async(backend, AsyncQueueState(2), 5, 4)
    assert len(backend.shipped) == (1 if via_poll else 2)


@pytest.mark.parametrize("via_poll", [False, True])
async def test_duplicate_response_ids_fail_closed(via_poll):
    backend = Backend()

    async def duplicate(job):
        return make_cohort(job["version"], "warmup")

    if via_poll:
        backend.poll_deferred = duplicate
    else:
        backend.wait_deferred = duplicate
    with pytest.raises(ValueError, match="duplicated or reused"):
        await run_async(backend, AsyncQueueState(2), 5, 4)


@pytest.mark.parametrize("job", [{}, {"export": object()}, {"export": float("nan")}, []])
async def test_nondurable_jobs_are_rejected_before_training(job):
    backend = Backend()

    async def invalid(version):
        return job

    backend.submit_deferred = invalid
    with pytest.raises(ValueError, match="serializable dictionary"):
        await run_async(backend, AsyncQueueState(2), 5, 4)
    assert not backend.shipped and not backend.receipts


async def test_worker_failure_does_not_substitute_cohorts():
    backend = Backend()

    async def fail(job):
        raise RuntimeError("worker failed")

    backend.wait_deferred = fail
    state = AsyncQueueState(2)
    with pytest.raises(RuntimeError, match="worker failed"):
        await run_async(backend, state, 5, 4)
    assert state.completed_steps == 2
    assert len(backend.shipped) == 2
    assert 0 in state.jobs


async def test_rejects_synchronous_state_and_changed_horizon():
    with pytest.raises(TypeError, match="asynchronous queue checkpoint"):
        await run_async(Backend(), QueueState(2), 5, 4)
    with pytest.raises(ValueError, match="another training horizon"):
        await run_async(Backend(), AsyncQueueState(2, generation_stop=7), 5, 4)


@pytest.mark.parametrize("corruption", ["missing", "overlap", "accounting", "response_ids"])
async def test_rejects_corrupt_checkpoint_state(corruption):
    backend = Backend(stop_at=2, ready={1: 2})
    state = AsyncQueueState(4)
    with pytest.raises(RuntimeError, match="interrupted"):
        await run_async(backend, state, 9, 4)
    if corruption == "missing":
        del state.jobs[0]
    elif corruption == "overlap":
        state.jobs[1] = copy.deepcopy(state.jobs[0])
    elif corruption == "accounting":
        state.generated_cohorts += 1
    else:
        state.generated_ids.clear()
    with pytest.raises(ValueError):
        state.validate()
