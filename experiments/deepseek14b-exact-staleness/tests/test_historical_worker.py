import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import msgspec
import pytest

from deepseek_study.rollouts.history_store import freeze_export, load_result, publish_job
from deepseek_study.runtime import historical_worker as worker


@pytest.mark.parametrize(
    "state,alive",
    [
        ("RUNNING", True),
        ("PENDING", True),
        ("COMPLETING", False),
        ("", False),
        ("FAILED", False),
        ("PREEMPTED", False),
    ],
)
def test_scheduler_watch_does_not_keep_failed_or_completing_learner_alive(monkeypatch, state, alive):
    monkeypatch.setattr(worker.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=state))
    assert worker.learner_alive(123) is alive


def test_scheduler_query_error_is_not_reported_as_healthy(monkeypatch):
    monkeypatch.setattr(worker.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""))
    with pytest.raises(RuntimeError, match="scheduler state"):
        worker.learner_alive(123)


@pytest.mark.parametrize("fails", [False, True])
async def test_manual_receiver_is_local_and_factory_restored_after_setup(fails):
    original = object()
    module = SimpleNamespace(setup_weight_receiver=original)

    async def setup():
        receiver = module.setup_weight_receiver()
        await receiver.initialize()
        await receiver.sync_startup(0, 1)
        with pytest.raises(ValueError, match="pinned initial model"):
            await receiver.sync_startup(4, 1)
        if fails:
            raise RuntimeError("setup failed")

    orch = SimpleNamespace(setup=setup)
    if fails:
        with pytest.raises(RuntimeError, match="setup failed"):
            await worker.setup_manual_receiver(module, orch)
    else:
        await worker.setup_manual_receiver(module, orch)
    assert module.setup_weight_receiver is original


async def test_partial_orchestrator_setup_has_safe_teardown_defaults():
    orch = SimpleNamespace()

    async def stop():
        orch.sender.close()
        assert orch.dispatcher is orch.watcher is orch.periodic_logger is orch.train_envs is None

    orch.stop = stop
    await worker.stop_orchestrator(orch)


@pytest.fixture
def session():
    return {
        "run_uuid": "run-a",
        "config_sha256": "config-a",
        "identity": {"sha256": "source-a"},
        "learner_job_id": "123",
        "contract": {"max_steps": 1000, "lag": 256},
    }


@pytest.mark.parametrize("field,value", [("learner_job_id", "other"), ("identity", {}), ("run_uuid", "")])
def test_worker_session_cannot_attach_to_another_run(session, study, monkeypatch, field, value):
    from deepseek_study.rollouts import historical

    monkeypatch.setattr(historical, "contract", lambda study: session["contract"])
    identity = session["identity"]
    worker.validate_session(study, session, 123, identity)
    session[field] = value
    with pytest.raises(ValueError):
        worker.validate_session(study, session, 123, identity)


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_uuid", "other"),
        ("config_sha256", "other"),
        ("identity_sha256", "other"),
        ("policy_version", True),
        ("policy_version", -1),
        ("policy_version", 744),
        ("consumption_step", 256),
    ],
)
def test_jobs_cannot_cross_session_or_exact_staleness_schedule(session, field, value):
    metadata = {
        "run_uuid": "run-a",
        "config_sha256": "config-a",
        "contract": session["contract"],
        "identity_sha256": "source-a",
        "worker_pool": "remote",
        "policy_version": 0,
        "consumption_step": 257,
    }
    job = {"job_id": "job-a", "metadata": metadata}
    assert worker.validate_session_job(job, session, Path("job-a")) == 0
    metadata[field] = value
    with pytest.raises(ValueError):
        worker.validate_session_job(job, session, Path("job-a"))


async def test_learner_exit_cancels_inflight_generation(monkeypatch):
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def work():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(worker, "learner_alive", lambda job_id: False)
    await asyncio.wait_for(worker.supervise_learner(work(), 123, interval=0), 1)
    assert started.is_set() and cancelled.is_set()


async def test_learner_scheduler_failure_cancels_work_and_propagates(monkeypatch):
    cancelled = asyncio.Event()

    async def work():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    def failed(job_id):
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr(worker, "learner_alive", failed)
    with pytest.raises(RuntimeError, match="scheduler unavailable"):
        await worker.supervise_learner(work(), 123)
    assert cancelled.is_set()


@pytest.fixture
def pending(study, tmp_path):
    source = tmp_path / "weights"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"test frozen weights")
    job = freeze_export(
        source,
        tmp_path / "export",
        {"run_uuid": "run-a", "config_sha256": "config-a", "policy_version": 0, "worker_pool": "remote"},
    )
    directory = publish_job(tmp_path / "export", tmp_path / "shared")
    study.output_dir.mkdir()
    (study.output_dir / "rollouts").mkdir()
    local = study.output_dir / "jobs"
    local.mkdir()
    notify = AsyncMock()
    orch = SimpleNamespace(
        admin_clients=SimpleNamespace(clients=["worker-local-pool"]),
        policy=SimpleNamespace(version=99),
        progress=SimpleNamespace(step=100),
        watcher=SimpleNamespace(ckpt_step=99, _notify_update=notify),
    )
    backend = SimpleNamespace(orch=orch, assert_idle=lambda: None)

    async def generate(version, purpose):
        assert version == orch.policy.version == orch.watcher.ckpt_step == 0
        assert purpose == "deferred" and orch.progress.step == 1
        cohort = SimpleNamespace(digest="payload-digest", output_tokens=5, response_ids=("r1", "r2"))
        archive = {
            "format": 2,
            "behavior_version": version,
            "purpose": purpose,
            "response_ids": list(cohort.response_ids),
            "payload": {"samples": b"samples"},
        }
        (study.output_dir / "rollouts" / "0-deferred-payload-digest.msgpack").write_bytes(
            msgspec.msgpack.encode(archive)
        )
        with (study.output_dir / "grading.jsonl").open("a") as stream:
            for response in cohort.response_ids:
                stream.write(json.dumps({"response_id": response, "grading": {"status": "ok"}}) + "\n")
        return cohort

    backend.generate = AsyncMock(side_effect=generate)
    return study, backend, directory, job, local


async def test_worker_loads_frozen_local_weights_and_publishes_auditable_result(pending, monkeypatch):
    from prime_rl.orchestrator import clients

    update = AsyncMock()
    monkeypatch.setattr(clients, "update_weights", update)
    _, backend, directory, job, local = pending
    assert await worker.process_job(*pending)
    update.assert_awaited_once_with(["worker-local-pool"], local / job["job_id"] / "weights", step=0)
    backend.orch.watcher._notify_update.assert_awaited_once_with(0)
    archive = msgspec.msgpack.decode(load_result(directory, job))
    assert [row["response_id"] for row in archive["grading_records"]] == ["r1", "r2"]
    assert json.loads((directory / "result.json").read_text())["metrics"]["output_tokens"] == 5
    assert not (local / job["job_id"]).exists()
    assert not await worker.process_job(*pending)
    assert backend.generate.await_count == 1


async def test_weight_load_failure_never_generates_or_retries(pending, monkeypatch):
    from prime_rl.orchestrator import clients

    monkeypatch.setattr(clients, "update_weights", AsyncMock(side_effect=RuntimeError("load failed")))
    _, backend, directory, _, _ = pending
    with pytest.raises(RuntimeError, match="load failed"):
        await worker.process_job(*pending)
    backend.generate.assert_not_called()
    assert not (directory / "result.json").exists()
    assert json.loads((directory / "failure.json").read_text())["error"] == "load failed"
    with pytest.raises(RuntimeError, match="worker failed"):
        await worker.process_job(*pending)


async def test_incomplete_prior_attempt_is_not_resampled(pending):
    _, backend, directory, _, _ = pending
    (directory / "attempt.json").write_text("{}")
    with pytest.raises(RuntimeError, match="explicit recovery"):
        await worker.process_job(*pending)
    backend.generate.assert_not_called()


async def test_cleanup_failure_does_not_invalidate_published_result(pending, monkeypatch):
    from prime_rl.orchestrator import clients

    _, _, directory, job, local = pending
    original = worker.shutil.rmtree

    def remove(path, *args, **kwargs):
        if Path(path) == local / job["job_id"]:
            raise OSError("temporary cleanup error")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(clients, "update_weights", AsyncMock())
    monkeypatch.setattr(worker.shutil, "rmtree", remove)
    assert await worker.process_job(*pending)
    assert load_result(directory, job) is not None
    assert not (directory / "failure.json").exists()


@pytest.mark.parametrize("lag,max_steps,count", [(256, 1000, 256), (256, 257, 1), (8, 1000, 8)])
def test_remote_worker_budget_ends_after_bootstrap_cohorts(lag, max_steps, count):
    assert worker.remote_versions({"lag": lag, "max_steps": max_steps}) == set(range(count))


async def test_remote_worker_cannot_claim_local_cohort(pending):
    _, backend, directory, job, _ = pending
    job["metadata"]["worker_pool"] = "local"
    with pytest.raises(ValueError, match="cannot claim a local"):
        await worker.process_job(*pending)
    assert not (directory / "attempt.json").exists()
    backend.generate.assert_not_called()


async def test_remote_controller_releases_its_pool_after_assigned_cohorts(study, tmp_path, monkeypatch):
    from prime_rl import monitors

    from deepseek_study.rollouts import controller, historical
    from deepseek_study.runtime import identity
    from deepseek_study.runtime.build import resolve

    study = study.model_copy(update={"lag": 2, "max_steps": 5, "historical_rollouts": tmp_path / "shared"})
    study.output_dir.mkdir()
    study.dataset_path.write_bytes(b"dataset")
    study.data_manifest.write_bytes(b"manifest")
    settings = historical.contract(study)
    source = {"sha256": "source-a"}
    session = {
        "contract": settings,
        "identity": source,
        "run_uuid": "run-a",
        "config_sha256": "learner-config",
        "learner_job_id": "123",
    }
    study.historical_rollouts.mkdir()
    (study.historical_rollouts / "session.json").write_text(json.dumps(session))
    for version in range(2):
        job_id = f"job-{version}"
        directory = study.historical_rollouts / "jobs" / job_id
        directory.mkdir(parents=True)
        metadata = {
            "contract": settings,
            "run_uuid": "run-a",
            "config_sha256": "learner-config",
            "identity_sha256": "source-a",
            "policy_version": version,
            "consumption_step": version + study.lag + 1,
            "worker_pool": "remote",
        }
        (directory / "job.json").write_text(json.dumps({"job_id": job_id, "metadata": metadata}))
    config = tmp_path / "orchestrator.json"
    config.write_text(resolve(study).orchestrator.model_dump_json())
    stopped = asyncio.Event()
    dispatcher = SimpleNamespace(start=stopped.wait, stop=AsyncMock(side_effect=stopped.set))
    orch = SimpleNamespace(setup=AsyncMock(), stop=AsyncMock(), dispatcher=dispatcher)
    module = SimpleNamespace(Orchestrator=lambda config: orch, setup_weight_receiver=object())
    monkeypatch.setitem(sys.modules, "prime_rl.orchestrator.orchestrator", module)
    monkeypatch.setattr(controller, "PrimeBackend", lambda study, orch: SimpleNamespace())
    monkeypatch.setattr(identity, "capture", lambda *args: source)
    monkeypatch.setattr(identity, "read_identity", lambda *args: source)
    monkeypatch.setattr(worker, "learner_alive", lambda job: True)
    process = AsyncMock(return_value=True)
    monkeypatch.setattr(worker, "process_job", process)
    monkeypatch.setattr(monitors, "finalize", AsyncMock())
    await asyncio.wait_for(worker.consume(study, config, 123), 2)
    assert process.await_count == 2
    assert stopped.is_set()
    orch.stop.assert_awaited_once()
    receipt = json.loads((study.output_dir / "historical-complete.json").read_text())
    assert receipt["completed_versions"] == [0, 1]
    assert receipt["learner_job_id"] == 123
