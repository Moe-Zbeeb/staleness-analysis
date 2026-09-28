import asyncio
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from deepseek_study.config import StudyConfig
from deepseek_study.rollouts import async_queue, controller, historical
from deepseek_study.rollouts.async_queue import AsyncQueueState
from deepseek_study.rollouts.queue import QueueState
from deepseek_study.runtime import checkpoints
from deepseek_study.runtime.build import build, resolve
from deepseek_study.runtime.deployment import RemoteInference


def historical_study(study, tmp_path):
    return StudyConfig.model_validate(
        study.model_dump() | {"historical_rollouts": tmp_path / "historical", "inference_tensor_parallel": 1}
    )


def test_historical_config_uses_filesystem_transport_for_every_component(study, tmp_path):
    configured = historical_study(study, tmp_path)
    configuration = build(configured, tmp_path / "configs")
    assert configuration.deployment.type == "single_node"
    assert configuration.deployment.num_train_gpus == study.trainer_gpus
    assert configuration.deployment.num_infer_gpus == study.inference_gpus
    for component in (configuration, configuration.trainer, configuration.orchestrator, configuration.inference):
        assert component.weight_broadcast.type == "filesystem"
        assert "host" not in component.weight_broadcast.model_dump()
        assert "port" not in component.weight_broadcast.model_dump()
    for name in ("trainer", "orchestrator", "inference"):
        written = json.loads((tmp_path / "configs" / f"{name}.json").read_text())
        assert written["weight_broadcast"]["type"] == "filesystem"
    protocol = json.loads((tmp_path / "configs/protocol.json").read_text())
    assert protocol["generation_method"] == "asynchronous_historical_weights"
    assert protocol["warmup_age"] == 0
    assert protocol["steady_state_age"] == study.lag
    assert protocol["warmup_updates"] == study.lag
    assert protocol["budget_includes_bootstrap"]
    assert protocol["responses_per_update"] == study.response_batch_size


def test_historical_transport_preserves_learner_and_scientific_settings(study, tmp_path):
    study = study.model_copy(update={"inference_tensor_parallel": 1})
    original = resolve(study).model_dump()
    changed = resolve(historical_study(study, tmp_path)).model_dump()
    original.pop("weight_broadcast")
    changed.pop("weight_broadcast")
    for name in ("trainer", "orchestrator", "inference"):
        original[name].pop("weight_broadcast")
        changed[name].pop("weight_broadcast")
    original["orchestrator"]["model"]["client"].pop("admin_base_url")
    changed["orchestrator"]["model"]["client"].pop("admin_base_url")
    assert original == changed


def test_historical_response_cap_propagates_to_training_and_inference(study, tmp_path):
    configured = StudyConfig.model_validate(
        study.model_dump()
        | {
            "historical_rollouts": tmp_path / "historical",
            "inference_tensor_parallel": 1,
            "prompt_max_tokens": 2048,
            "response_max_tokens": 6144,
        }
    )
    configuration = resolve(configured)
    assert configuration.trainer.model.seq_len == 8192
    assert configuration.inference.vllm.max_model_len == 8192
    assert configuration.orchestrator.train.sampling.max_completion_tokens == 6144
    assert configuration.orchestrator.eval is None
    assert configuration.orchestrator.batch_size == study.response_batch_size
    assert configuration.orchestrator.train.filter_zero_advantages is False


def test_historical_workers_cannot_join_live_remote_inference(study, tmp_path):
    remote = RemoteInference(
        trainer_host="learner",
        router_url="http://learner:8000/v1",
        worker_urls=[f"http://worker:{8100 + index}/v1" for index in range(study.inference_gpus)],
        hardware=[],
    )
    with pytest.raises(ValueError, match="must not join"):
        resolve(historical_study(study, tmp_path), remote=remote)


@pytest.mark.parametrize("changes", [{"historical_rollouts": Path("relative")}, {"lag": 0}])
def test_historical_mode_requires_positive_lag_and_absolute_storage(study, tmp_path, changes):
    with pytest.raises(ValidationError):
        StudyConfig.model_validate(study.model_dump() | {"historical_rollouts": tmp_path / "historical"} | changes)


def test_legacy_fingerprint_is_unchanged_when_historical_mode_is_disabled(study):
    legacy_values = study.model_dump(mode="json", exclude={"output_dir", "metrics_mirror_root", "historical_rollouts"})
    expected = hashlib.sha256(json.dumps(legacy_values, sort_keys=True).encode()).hexdigest()
    assert study.fingerprint() == expected


def test_historical_mode_is_part_of_checkpoint_identity(study, tmp_path):
    historical_config = historical_study(study, tmp_path)
    assert historical_config.fingerprint() != study.fingerprint()
    other = StudyConfig.model_validate(study.model_dump() | {"historical_rollouts": tmp_path / "other-history"})
    assert historical_config.fingerprint() != other.fingerprint()


def make_checkpoint(directory):
    files = (
        "trainer/.metadata",
        "trainer/rank0.distcp",
        "orchestrator/progress.pt",
        "rng/rank_0.pt",
        "historical/job/job.json",
        "historical/job/weights/model.safetensors",
    )
    for name in files:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original")
    state = AsyncQueueState(2, completed_steps=1, generated_cohorts=1, jobs={0: {"export": "immutable"}})
    digest = checkpoints.seal(directory, 1)
    checkpoints.save(directory / "study", state, "config", "identity", digest)
    return state


@pytest.mark.parametrize("component", ["job.json", "weights/model.safetensors"])
def test_historical_checkpoint_files_are_hashed_and_validated(tmp_path, component):
    state = make_checkpoint(tmp_path)
    checkpoints.verify_components(tmp_path)
    restored = checkpoints.load(tmp_path / "study", "config", "identity")
    assert isinstance(restored, AsyncQueueState)
    assert restored.jobs == state.jobs
    manifest = json.loads((tmp_path / "study/components.json").read_text())
    record = next(row for row in manifest if row["path"] == f"historical/job/{component}")
    assert record["sha256"] == hashlib.sha256(b"original").hexdigest()
    (tmp_path / "historical/job" / component).write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        checkpoints.verify_components(tmp_path)


def test_incomplete_historical_checkpoint_cannot_pass_validation(tmp_path):
    make_checkpoint(tmp_path)
    (tmp_path / "historical/job/weights/model.safetensors").unlink()
    with pytest.raises(ValueError, match="missing or incomplete"):
        checkpoints.verify_components(tmp_path)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("resuming", [False, True])
async def test_controller_routes_fresh_and_resumed_state_to_correct_scheduler(study, tmp_path, monkeypatch, enabled, resuming):
    if enabled:
        study = historical_study(study, tmp_path)
    config_directory = tmp_path / "config"
    build(study, config_directory)
    resumed_state = (AsyncQueueState if enabled else QueueState)(study.lag, completed_steps=3)
    events = []
    stopped = asyncio.Event()

    class Dispatcher:
        async def start(self):
            await stopped.wait()

        async def stop(self):
            stopped.set()

    class Orchestrator:
        def __init__(self, config):
            self.dispatcher = Dispatcher()
            self.policy = SimpleNamespace(version=3 if resuming else 0)
            self.progress = SimpleNamespace(step=4 if resuming else 1)

        async def setup(self):
            events.append("setup")

        async def stop(self):
            events.append("stop")

    class Backend:
        def __init__(self, config, orchestrator):
            events.append("backend")

        async def restore_jobs(self, state, checkpoint):
            assert isinstance(state, AsyncQueueState)
            assert checkpoint == (tmp_path / "resume" if resuming else None)
            events.append("restore_jobs")

        async def close(self):
            events.append("close")

    async def runner(backend, state, max_steps, responses):
        assert isinstance(state, AsyncQueueState) == enabled
        assert (state is resumed_state) == resuming
        assert responses == study.response_batch_size
        state.completed_steps = max_steps
        events.append("run")

    async def finalize():
        events.append("finalize")

    monkeypatch.setitem(sys.modules, "prime_rl.orchestrator.orchestrator", SimpleNamespace(Orchestrator=Orchestrator))
    monkeypatch.setattr(controller, "read_identity", lambda path: {"sha256": "identity"})
    monkeypatch.setattr(controller.checkpoints, "verify_components", lambda path: events.append("verify"))
    monkeypatch.setattr(controller.checkpoints, "load", lambda *args: resumed_state)
    monkeypatch.setattr(historical, "HistoricalBackend", Backend)
    monkeypatch.setattr(controller, "PrimeBackend", Backend)
    monkeypatch.setattr(async_queue, "run_async", runner)
    monkeypatch.setattr(controller, "run", runner)
    monkeypatch.setattr(controller.monitors, "finalize", finalize)
    await controller.control(study, config_directory / "orchestrator.json", tmp_path / "resume" if resuming else None)
    assert events[-1] == "stop"
    assert ("restore_jobs" in events) == enabled
    assert ("close" in events) == enabled
    assert ("verify" in events) == resuming
    completion = json.loads((study.output_dir / "study-complete.json").read_text())
    assert completion["step"] == study.max_steps
    assert completion["lag"] == study.lag


async def test_controller_always_stops_orchestrator_when_historical_close_raises(study, tmp_path, monkeypatch):
    study = historical_study(study, tmp_path)
    config_directory = tmp_path / "config"
    build(study, config_directory)
    events = []

    class Orchestrator:
        def __init__(self, config):
            self.policy = SimpleNamespace(version=0)
            self.progress = SimpleNamespace(step=1)

        async def setup(self):
            pass

        async def stop(self):
            events.append("stop")

    class Backend:
        def __init__(self, study, orch):
            pass

        async def restore_jobs(self, state, checkpoint):
            raise RuntimeError("restore failed")

        async def close(self):
            raise RuntimeError("close failed")

    monkeypatch.setitem(sys.modules, "prime_rl.orchestrator.orchestrator", SimpleNamespace(Orchestrator=Orchestrator))
    monkeypatch.setattr(controller, "read_identity", lambda path: {"sha256": "identity"})
    monkeypatch.setattr(historical, "HistoricalBackend", Backend)
    with pytest.raises(RuntimeError, match="close failed"):
        await controller.control(study, config_directory / "orchestrator.json")
    assert events == ["stop"]


@pytest.mark.parametrize("asset", ["model_path", "dataset_path", "prepared_model_path", "data_manifest"])
def test_historical_storage_cannot_overlap_input_assets(study, asset):
    path = getattr(study, asset)
    with pytest.raises(ValidationError, match="overlap input assets"):
        StudyConfig.model_validate(study.model_dump() | {"historical_rollouts": path})
    with pytest.raises(ValidationError, match="overlap input assets"):
        StudyConfig.model_validate(study.model_dump() | {"historical_rollouts": path / "history"})


@pytest.mark.parametrize("relation", ["same", "child", "parent"])
def test_historical_storage_cannot_overlap_active_run(study, tmp_path, relation):
    study.output_dir = tmp_path / "active" / "run"
    paths = {"same": study.output_dir, "child": study.output_dir / "history", "parent": study.output_dir.parent}
    with pytest.raises(ValidationError, match="active run directory must be separate"):
        StudyConfig.model_validate(study.model_dump() | {"historical_rollouts": paths[relation]})


def test_historical_storage_cannot_overlap_metric_mirror(study, tmp_path):
    metrics = tmp_path / "metrics"
    with pytest.raises(ValidationError, match="metric mirror must be separate"):
        StudyConfig.model_validate(
            study.model_dump() | {"metrics_mirror_root": metrics, "historical_rollouts": metrics / study.output_dir.name}
        )


@pytest.mark.parametrize("enabled", [False, True])
async def test_queue_mode_mismatch_is_rejected_before_orchestrator_setup(study, tmp_path, monkeypatch, enabled):
    if enabled:
        study = historical_study(study, tmp_path)
    config_directory = tmp_path / "config"
    build(study, config_directory)
    state = (QueueState if enabled else AsyncQueueState)(study.lag)

    def forbidden(config):
        raise AssertionError("Mismatched recovery reached orchestrator initialization")

    monkeypatch.setitem(sys.modules, "prime_rl.orchestrator.orchestrator", SimpleNamespace(Orchestrator=forbidden))
    monkeypatch.setattr(controller, "read_identity", lambda path: {"sha256": "identity"})
    monkeypatch.setattr(controller.checkpoints, "verify_components", lambda path: None)
    monkeypatch.setattr(controller.checkpoints, "load", lambda *args: state)
    with pytest.raises(ValueError, match="queue mode differs"):
        await controller.control(study, config_directory / "orchestrator.json", tmp_path / "resume")
