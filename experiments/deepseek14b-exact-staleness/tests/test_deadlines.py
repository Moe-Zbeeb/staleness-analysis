import asyncio
from types import SimpleNamespace

import pytest

from deepseek_study.rollouts.controller import PrimeBackend
from deepseek_study.runtime.build import resolve


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
