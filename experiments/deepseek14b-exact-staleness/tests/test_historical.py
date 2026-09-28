import asyncio
import copy
import json
import shutil
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import msgspec
import pytest
from prime_rl.transports.batch import TrainingSample

from deepseek_study.rollouts.async_queue import AsyncQueueState
from deepseek_study.rollouts.controller import Payload, PrimeBackend
from deepseek_study.rollouts.historical import HistoricalBackend, contract, decode_cohort, worker_pool
from deepseek_study.rollouts.history_store import freeze_export, pin_job, publish_job, publish_result, validate_job


def make_archive(study, version=0):
    count = study.response_batch_size
    ids = tuple(f"response-{index}" for index in range(count))
    questions = tuple(f"question-{index // study.responses_per_prompt}" for index in range(count))
    samples = [
        TrainingSample(
            token_ids=[1, 2, 3, 4],
            mask=[False, False, True, True],
            logprobs=[0.0, 0.0, -0.3, -0.2],
            temperatures=[1.0] * 4,
            advantages=[0.0, 0.0, -0.5, -0.5],
            env_name="cleaned-deepscaler",
        )
        for _ in range(count)
    ]
    payload = Payload(
        samples=msgspec.msgpack.encode(samples),
        policy_spans=((version, version),) * count,
        rewards=(0.0,) * count,
        truncated=(False,) * count,
        task_keys=questions,
        sample_response_ids=tuple(reversed(ids)),
        sample_task_keys=tuple(reversed(questions)),
    )
    archive = {
        "format": 2,
        "behavior_version": version,
        "purpose": "deferred",
        "response_ids": ids,
        "payload": payload,
    }
    return archive


@pytest.fixture
def historical_setup(study, tmp_path):
    study.dataset_path.write_bytes(b"dataset")
    study.data_manifest.write_bytes(b"manifest")
    study.output_dir.mkdir()
    backend = HistoricalBackend.__new__(HistoricalBackend)
    backend.study = study
    backend.root = tmp_path / "shared"
    backend.local = study.output_dir / "historical-exports"
    backend.local.mkdir()
    backend.contract = contract(study)
    backend.run_uuid = "original-run"
    backend.identity = {"sha256": "source-identity"}
    backend.registered_jobs = {}
    backend.publications = {}
    backend.local_generations = {}
    backend.accepted = set()
    backend.export_seconds = 0.0
    backend.historical_wait_seconds = 0.0
    archive = make_archive(study)
    backend.source = SimpleNamespace(expected_questions=lambda *args: Counter(archive["payload"].task_keys))
    source = tmp_path / "export"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"immutable-policy-weights")
    metadata = {
        "run_uuid": backend.run_uuid,
        "config_sha256": study.fingerprint(),
        "identity_sha256": backend.identity["sha256"],
        "policy_version": 0,
        "consumption_step": study.lag + 1,
        "contract": backend.contract,
        "worker_pool": "remote",
    }
    job = freeze_export(source, backend.local / "step_0", metadata)
    backend.register_job(0, job)
    return backend, job, archive


def publish(backend, job, archive, **metrics):
    directory = publish_job(backend.local / "step_0", backend.root)
    publish_result(
        directory,
        job,
        msgspec.msgpack.encode(archive),
        {"output_tokens": backend.study.response_batch_size * 2, "generation_wall_seconds": 2.5, **metrics},
    )
    return directory


def test_decode_uses_response_masks_and_preserves_sample_provenance(historical_setup):
    backend, job, archive = historical_setup
    cohort = decode_cohort(msgspec.msgpack.encode(archive), job)
    assert cohort.output_tokens == backend.study.response_batch_size * 2
    assert cohort.response_ids == archive["response_ids"]
    assert cohort.payload.sample_response_ids == tuple(reversed(cohort.response_ids))
    assert cohort.payload == archive["payload"]


@pytest.mark.parametrize("corruption", ["response_ids", "mapping", "spans", "rewards", "timeout_count", "samples"])
def test_decode_rejects_inconsistent_provenance(historical_setup, corruption):
    _, job, archive = historical_setup
    payload = archive["payload"]
    if corruption == "response_ids":
        archive["response_ids"] = ("duplicate",) * len(archive["response_ids"])
    elif corruption == "mapping":
        archive["payload"] = replace(payload, sample_task_keys=payload.task_keys)
    elif corruption == "spans":
        archive["payload"] = replace(payload, policy_spans=((0, 1),) * len(payload.policy_spans))
    elif corruption == "rewards":
        archive["payload"] = replace(payload, rewards=(float("nan"),) * len(payload.rewards))
    elif corruption == "timeout_count":
        archive["payload"] = replace(payload, verification_timeouts=len(payload.rewards) + 1)
    else:
        archive["payload"] = replace(payload, samples=msgspec.msgpack.encode([]))
    with pytest.raises(ValueError):
        decode_cohort(msgspec.msgpack.encode(archive), job)


@pytest.mark.parametrize(
    "field,value",
    [
        ("logprobs", [0.0]),
        ("logprobs", [0.0, 0.0, float("inf"), -1.0]),
        ("advantages", None),
        ("advantages", [0.0, 0.0, float("nan"), 0.0]),
        ("temperatures", [0.9] * 4),
        ("mask", [False] * 4),
        ("token_ids", [1, 2, 3, -1]),
        ("rl_weights", [0.0] * 4),
        ("ce_weights", [1.0] * 4),
        ("ref_kl_weights", [1.0] * 4),
    ],
)
def test_decode_rejects_invalid_training_streams(historical_setup, field, value):
    _, job, archive = historical_setup
    samples = msgspec.msgpack.decode(archive["payload"].samples, type=list[TrainingSample])
    setattr(samples[0], field, value)
    archive["payload"] = replace(archive["payload"], samples=msgspec.msgpack.encode(samples))
    with pytest.raises(ValueError):
        decode_cohort(msgspec.msgpack.encode(archive), job)


@pytest.mark.parametrize("policy,consumption", [(1, 33), (True, 34), (-1, 32), (38, 71)])
def test_decode_rejects_invalid_version_mapping(historical_setup, policy, consumption):
    _, job, archive = historical_setup
    job = copy.deepcopy(job)
    job["metadata"]["policy_version"] = policy
    job["metadata"]["consumption_step"] = consumption
    with pytest.raises(ValueError):
        decode_cohort(msgspec.msgpack.encode(archive), job)


async def test_poll_cross_checks_tokens_and_does_not_allow_metric_provenance_override(historical_setup):
    backend, job, archive = historical_setup
    publish(backend, job, archive, policy_version=99, consumption_step=999, digest="wrong")
    cohort = await backend.poll_deferred(job)
    assert cohort.output_tokens == backend.study.response_batch_size * 2
    again = await backend.poll_deferred(job)
    assert again == cohort
    rows = [json.loads(line) for line in (backend.study.output_dir / "generations.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["policy_version"] == 0
    assert rows[0]["consumption_step"] == backend.study.lag + 1
    assert rows[0]["digest"] == cohort.digest


@pytest.mark.parametrize("metrics", [{"output_tokens": 1}, {"output_tokens": 16.0}, {"generation_wall_seconds": -1}])
async def test_poll_rejects_invalid_receipt_metrics(historical_setup, metrics):
    backend, job, archive = historical_setup
    publish(backend, job, archive, **metrics)
    with pytest.raises(ValueError):
        await backend.poll_deferred(job)
    assert not backend.accepted


async def test_poll_rejects_unregistered_or_changed_job(historical_setup):
    backend, job, _ = historical_setup
    backend.registered_jobs.clear()
    with pytest.raises(ValueError, match="submitted or restored"):
        await backend.poll_deferred(job)
    backend.register_job(0, job)
    changed = copy.deepcopy(job)
    changed["metadata"]["identity_sha256"] = "another-source"
    with pytest.raises(ValueError, match="scientific identity"):
        await backend.poll_deferred(changed)


async def test_resume_loads_completed_result_after_shared_weights_were_deleted(historical_setup, tmp_path):
    backend, job, archive = historical_setup
    shared = publish(backend, job, archive)
    checkpoint = tmp_path / "checkpoint"
    pin_job(backend.local / "step_0", checkpoint / "historical" / job["job_id"])
    shutil.rmtree(shared / "weights")
    shutil.rmtree(backend.local / "step_0")
    backend.registered_jobs.clear()
    backend.run_uuid = "resumed-run"
    state = AsyncQueueState(backend.study.lag, jobs={0: job})
    await backend.restore_jobs(state, checkpoint)
    assert not backend.publications
    validate_job(backend.local / "step_0", job)
    cohort = await backend.poll_deferred(job)
    assert cohort.response_ids == archive["response_ids"]


async def test_resume_republishes_unfinished_job_from_pinned_export(historical_setup, tmp_path):
    backend, job, _ = historical_setup
    checkpoint = tmp_path / "checkpoint"
    pin_job(backend.local / "step_0", checkpoint / "historical" / job["job_id"])
    shutil.rmtree(backend.local / "step_0")
    backend.registered_jobs.clear()
    await backend.restore_jobs(AsyncQueueState(backend.study.lag, jobs={0: job}), checkpoint)
    await backend.close()
    validate_job(backend.root / "jobs" / job["job_id"], job)
    assert await backend.poll_deferred(job) is None


async def test_checkpoint_pins_unfinished_exports_before_complete_marker(historical_setup, monkeypatch):
    backend, job, _ = historical_setup
    backend.study.checkpoint_first_step = True
    state = AsyncQueueState(backend.study.lag, completed_steps=1, jobs={0: job})
    destination = backend.study.output_dir / "checkpoints" / "step_1" / "historical" / job["job_id"]
    commits = []

    async def commit(self, state, receipt):
        validate_job(destination, job)
        commits.append(receipt)

    monkeypatch.setattr(PrimeBackend, "commit", commit)
    await backend.commit(state, {"step": 1})
    assert len(commits) == 1
    shutil.rmtree(backend.local / "step_0")
    validate_job(destination, job)


async def test_checkpoint_missing_export_cannot_be_marked_complete(historical_setup, monkeypatch):
    backend, job, _ = historical_setup
    backend.study.checkpoint_first_step = True
    state = AsyncQueueState(backend.study.lag, completed_steps=1, jobs={0: job})
    shutil.rmtree(backend.local / "step_0")
    commits = []

    async def commit(self, state, receipt):
        commits.append(receipt)

    monkeypatch.setattr(PrimeBackend, "commit", commit)
    with pytest.raises((ValueError, FileNotFoundError)):
        await backend.commit(state, {"step": 1})
    assert not commits


@pytest.mark.parametrize(
    "version,lag,pool", [(0, 256, "remote"), (255, 256, "remote"), (256, 256, "local"), (743, 256, "local")]
)
def test_generation_pool_assignment_is_version_bound(version, lag, pool):
    assert worker_pool(version, lag) == pool


def local_generation(backend, monkeypatch, gate=None, failure=None):
    version = backend.study.lag
    broadcast = backend.study.output_dir / "broadcasts" / f"step_{version}"
    broadcast.mkdir(parents=True)
    (broadcast / ".finished").touch()
    (broadcast / "model.safetensors").write_bytes(b"current-local-policy")
    started = asyncio.Event()

    async def generate(actual, purpose):
        assert actual == version and purpose == "deferred"
        started.set()
        if gate is not None:
            await gate.wait()
        if failure is not None:
            raise failure
        archive = make_archive(backend.study, version)
        metadata = {
            "policy_version": version,
            "consumption_step": version + backend.study.lag + 1,
            "contract": backend.contract,
        }
        raw = msgspec.msgpack.encode(archive)
        cohort = decode_cohort(raw, {"metadata": metadata})
        path = backend.study.output_dir / "rollouts" / f"{version}-deferred-{cohort.digest}.msgpack"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
        backend.append(
            "generations.jsonl", {"policy_version": version, "purpose": "deferred", "digest": cohort.digest}
        )
        return cohort

    generator = AsyncMock(side_effect=generate)
    monkeypatch.setattr(backend, "generate", generator)
    return version, started, generator


async def test_local_generation_overlaps_training_but_finishes_before_weight_change(historical_setup, monkeypatch):
    backend, _, _ = historical_setup
    gate = asyncio.Event()
    version, started, generator = local_generation(backend, monkeypatch, gate)
    synchronize = AsyncMock()
    monkeypatch.setattr(PrimeBackend, "synchronize", synchronize)
    job = await backend.submit_deferred(version)
    assert job["metadata"]["worker_pool"] == "local"
    assert not backend.publications
    await started.wait()
    assert await backend.poll_deferred(job) is None
    synchronizing = asyncio.create_task(backend.synchronize(version + 1))
    await asyncio.sleep(0)
    synchronize.assert_not_awaited()
    assert not synchronizing.done()
    gate.set()
    await synchronizing
    synchronize.assert_awaited_once_with(version + 1)
    cohort = await backend.poll_deferred(job)
    assert cohort.version == version
    assert not (backend.root / "jobs" / job["job_id"]).exists()
    assert (backend.local / f"step_{version}" / "result.json").is_file()
    rows = (backend.study.output_dir / "generations.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert await backend.poll_deferred(job) == cohort
    assert len((backend.study.output_dir / "generations.jsonl").read_text().splitlines()) == 1
    generator.assert_awaited_once()
    await backend.close()


async def test_failed_local_generation_blocks_weight_update_and_reroll(historical_setup, monkeypatch):
    backend, _, _ = historical_setup
    version, _, generator = local_generation(backend, monkeypatch, failure=RuntimeError("generation failed"))
    synchronize = AsyncMock()
    monkeypatch.setattr(PrimeBackend, "synchronize", synchronize)
    job = await backend.submit_deferred(version)
    with pytest.raises(RuntimeError, match="generation failed"):
        await backend.synchronize(version + 1)
    synchronize.assert_not_awaited()
    directory = backend.local / f"step_{version}"
    assert (directory / "failure.json").is_file()
    assert not (directory / "result.json").exists()
    with pytest.raises(RuntimeError, match="explicit recovery"):
        await backend.generate_local(job, directory)
    generator.assert_awaited_once()
    await backend.close()


async def test_local_jobs_must_be_materialized_before_checkpoint_or_recovery(historical_setup, monkeypatch, tmp_path):
    backend, _, _ = historical_setup
    version, _, _ = local_generation(backend, monkeypatch)
    job = await backend.submit_deferred(version)
    await backend.local_generations[version]
    state = AsyncQueueState(backend.study.lag, jobs={version: job})
    commit = AsyncMock()
    monkeypatch.setattr(PrimeBackend, "commit", commit)
    with pytest.raises(RuntimeError, match="materialized"):
        await backend.commit(state, {"step": version + 1})
    commit.assert_not_awaited()
    with pytest.raises(RuntimeError, match="unfinished local"):
        await backend.restore_jobs(state, tmp_path / "checkpoint")
    await backend.close()
