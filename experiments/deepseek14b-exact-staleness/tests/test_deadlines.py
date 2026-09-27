import asyncio
import threading
from collections import Counter
from types import SimpleNamespace

import msgspec
import pytest

from deepseek_study.rollouts.controller import Payload, PrimeBackend, payload_digest
from deepseek_study.rollouts.queue import Cohort
from deepseek_study.runtime.build import resolve
from prime_rl.transports.batch import TrainingSample


async def backend(training_delay, transfer_delay, training_timeout=0.1, transfer_timeout=0.01):
    instance = PrimeBackend.__new__(PrimeBackend)
    instance.assert_idle = lambda: None
    instance.study = SimpleNamespace(
        training_timeout_seconds=training_timeout, weight_transfer_timeout_seconds=transfer_timeout
    )
    policy = SimpleNamespace(version=0)
    watcher = SimpleNamespace(ckpt_step=0)

    async def wait_published(version):
        await asyncio.sleep(training_delay)

    async def apply(version):
        await asyncio.sleep(transfer_delay)
        watcher.ckpt_step = policy.version = version

    watcher.receiver = SimpleNamespace(wait_published=wait_published)
    watcher.apply_policy_update = apply
    instance.orch = SimpleNamespace(watcher=watcher, policy=policy)
    return instance


async def test_slow_learner_does_not_consume_the_weight_transfer_deadline():
    instance = await backend(0.03, 0)
    await instance.synchronize(1)
    assert instance.orch.policy.version == 1
    assert instance.training_wait_seconds >= 0.03


async def test_training_timeout_fails_without_advancing_policy():
    instance = await backend(0.1, 0, training_timeout=0.01)
    with pytest.raises(TimeoutError, match="training deadline"):
        await instance.synchronize(1)
    assert instance.orch.policy.version == 0


async def test_transfer_timeout_remains_separate_and_finite():
    instance = await backend(0, 0.1)
    with pytest.raises(TimeoutError, match="weight-transfer deadline"):
        await instance.synchronize(1)
    assert instance.orch.policy.version == 0


def test_broadcast_handshake_allows_full_cohort_generation(study):
    config = resolve(study)
    assert config.trainer.weight_broadcast.timeout >= study.generation_timeout_seconds
    assert study.training_timeout_seconds > study.weight_transfer_timeout_seconds


def shipping_backend(send, pack=lambda samples: samples, timeout=0.03):
    instance = PrimeBackend.__new__(PrimeBackend)
    instance.study = SimpleNamespace(
        dispatch_timeout_seconds=timeout, response_batch_size=2, prompts_per_update=1, responses_per_prompt=2, lag=8
    )
    instance.orch = SimpleNamespace(
        progress=SimpleNamespace(step=1, total_samples=0, total_problems=0, total_tokens=0),
        policy=SimpleNamespace(version=0),
        packer=SimpleNamespace(pack=pack),
        sender=SimpleNamespace(send=send),
    )
    instance.source = SimpleNamespace(expected_questions=lambda *_: Counter({"question": 2}))
    instance.shipped = None
    samples = [
        TrainingSample(
            token_ids=[1, 2],
            mask=[False, True],
            logprobs=[0.0, -1.0],
            temperatures=[1.0, 1.0],
            advantages=[0.0, advantage],
            env_name="test",
        )
        for advantage in (-1.0, 1.0)
    ]
    payload = Payload(
        msgspec.msgpack.encode(samples),
        ((0, 0), (0, 0)),
        (0.0, 1.0),
        (False, False),
        ("question", "question"),
        ("response-1", "response-2"),
        ("question", "question"),
    )
    cohort = Cohort(0, payload.sample_response_ids, 2, payload, payload_digest(payload))
    return instance, cohort


async def test_batch_ready_deadline_cancels_wait_without_advancing_the_clock():
    started, cancelled = asyncio.Event(), asyncio.Event()
    attempts = 0

    async def send(_):
        nonlocal attempts
        attempts += 1
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    instance, cohort = shipping_backend(send)
    with pytest.raises(TimeoutError, match="batch-dispatch deadline"):
        await instance.ship(cohort, learner_version=0)
    assert started.is_set() and cancelled.is_set()
    assert attempts == 1
    assert vars(instance.orch.progress) == {"step": 1, "total_samples": 0, "total_problems": 0, "total_tokens": 0}
    assert instance.orch.policy.version == 0
    assert instance.shipped is None


async def test_packing_deadline_never_sends_a_partial_batch():
    release = threading.Event()
    finished = threading.Event()
    sent = []

    def pack(samples):
        release.wait()
        finished.set()
        return samples

    async def send(grid):
        sent.append(grid)

    instance, cohort = shipping_backend(send, pack)
    try:
        with pytest.raises(TimeoutError, match="batch-dispatch deadline"):
            await instance.ship(cohort, learner_version=0)
        assert not sent
        assert instance.orch.progress.step == 1
        assert instance.shipped is None
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 1)


async def test_successful_batch_dispatch_advances_progress_only_after_send():
    ready, release = asyncio.Event(), asyncio.Event()
    received = []

    async def send(grid):
        ready.set()
        await release.wait()
        received.extend(grid)

    instance, cohort = shipping_backend(send, timeout=1)
    task = asyncio.create_task(instance.ship(cohort, learner_version=0))
    await asyncio.wait_for(ready.wait(), timeout=1)
    assert instance.orch.progress.step == 1
    assert instance.shipped is None
    release.set()
    await task
    assert len(received) == 2
    assert vars(instance.orch.progress) == {"step": 2, "total_samples": 2, "total_problems": 1, "total_tokens": 2}
    assert instance.orch.policy.version == 0
    assert instance.shipped is cohort
