import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from deepseek_study.runtime.build import build, resolve
from deepseek_study.config import StudyConfig
from deepseek_study.rollouts.controller import FiniteSource
from deepseek_study.recipe import baseline
from prime_rl.orchestrator.types import TaskRequest
from prime_rl.orchestrator.curriculum.samplers.standard import StandardSampler


def test_missing_scientific_settings_are_not_guessed():
    with pytest.raises(ValidationError):
        StudyConfig.model_validate({})


def test_official_config_and_taskset_plugin_resolve(study, tmp_path):
    config = build(study, tmp_path / "configs")
    assert config.trainer.model.optimization_dtype == "float32"
    assert config.orchestrator.num_train_workers == study.trainer_gpus
    assert config.orchestrator.train.filter_zero_advantages is False
    assert config.orchestrator.train.source[0].group_size == study.responses_per_prompt
    assert config.trainer.loss.import_path == "deepseek_study.learning.loss.clipped_grpo"
    assert config.inference.vllm.model == str(study.prepared_model_path)
    assert config.inference.vllm.logprobs_mode == "raw_logprobs"
    assert config.orchestrator.model.client.base_url.endswith(":8000/v1")
    serialized = json.loads((tmp_path / "configs" / "orchestrator.json").read_text())
    assert serialized["renderer"]["name"] == "default"


def test_finite_source_uses_official_request_contract_and_pins_generation_step():
    class Source:
        def next_task(self, *, step):
            return TaskRequest("test", object(), step)

    finite = FiniteSource(Source())
    finite.begin(7, 2, 4)
    assert finite.next_task(step=9).step == 8
    assert finite.next_task(step=9).step == 8
    assert finite.next_task(step=9) is None


def test_resume_paths_are_propagated(study, tmp_path):
    checkpoint = tmp_path / "previous" / "checkpoints" / "step_35"
    config = resolve(study, checkpoint)
    assert config.trainer.resume.dir == checkpoint
    assert config.orchestrator.resume.dir == checkpoint


@pytest.mark.parametrize("lag", [0, 1, 8, 32, 64])
@pytest.mark.parametrize("profile", ["80gb", "40gb"])
def test_baseline_profiles_resolve_without_changing_scientific_recipe(lag, profile):
    study = baseline(lag, profile)
    config = resolve(study)
    assert config.trainer.max_steps == 1000
    assert config.orchestrator.batch_size == 512
    assert config.orchestrator.max_off_policy_steps == lag
    assert config.trainer.scheduler.warmup_steps == 30
    assert config.trainer.scheduler.decay_steps == 0
    assert config.trainer.model.dp_replicate == 1
    assert config.inference.vllm.max_num_seqs == (2 if profile == "40gb" else 16)
    assert (config.trainer.model.ac_offloading is not None) == (profile == "40gb")
    assert config.trainer.model.fused_lm_head_token_chunk_size == (1024 if profile == "40gb" else 8192)


@pytest.mark.parametrize(
    "changes",
    [
        {"lag": -1},
        {"lag": 1.5},
        {"lag": True},
        {"lag": 70},
        {"learning_rate": float("inf")},
        {"reward_outer_timeout_seconds": 8},
    ],
)
def test_invalid_run_contracts_fail_before_launch(study, changes):
    with pytest.raises(ValidationError):
        StudyConfig.model_validate(study.model_dump() | changes)


@pytest.mark.parametrize("lag", [0, 1, 8, 32])
def test_prompt_assignment_depends_on_consumption_update_not_generation_order(lag):
    tasks = [SimpleNamespace(key=str(index), data=SimpleNamespace(question_id=str(index))) for index in range(17)]
    sampler = StandardSampler(tasks)

    class Source:
        curricula = {"data": SimpleNamespace(sampler=sampler, gates={})}

        def next_task(self, *, step):
            return TaskRequest("data", next(sampler), step)

    source = FiniteSource(Source())
    for cohort_index in [0, lag, 1, lag + 1, 4, lag + 4]:
        source.begin(0, 3, 8, cohort_index)
        actual = [source.next_task(step=1).task.data.question_id for _ in range(3)]
        expected = [str((cohort_index * 3 + index) % 17) for index in range(3)]
        assert actual == expected
        assert source.next_task(step=1) is None
        assert source.expected_questions(cohort_index, 3, 8) == {question: 8 for question in expected}
