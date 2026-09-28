import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deepseek_study.dataset import prepare as data
from deepseek_study.recipe import baseline


def test_prepared_full_dataset_has_explicit_accounted_exclusions():
    root = Path(__file__).resolve().parents[1]
    path = root / "assets/train-manifest-v4.json"
    if not path.is_file():
        pytest.skip("Full local data preparation has not been run")
    study = baseline(32)
    rows = data.prepared_rows(
        root / "assets/dataset-fixture/data/train.parquet",
        path,
        study.prompt_instruction,
        study.reward_timeout_seconds,
        study.prompt_max_tokens,
    )
    manifest = data.load_manifest(path)
    assert len(rows) == manifest["included_rows"]
    assert len(rows) + manifest["excluded_rows"] == 37713
    excluded = {record["id"] for record in manifest["records"] if not record["included"]}
    assert not ({row["id"] for row in rows} & excluded)
    assert all(record["prompt_tokens"] <= 2048 for record in manifest["records"] if record["included"])


def test_data_manifest_rejects_tampering_and_changed_grader(tmp_path):
    body = {"format": 1, "contract": data.data_contract("prompt", 128, 8)}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({**body, "sha256": data.digest(body)}))
    assert data.load_manifest(path)["contract"] == body["contract"]
    body["contract"]["reward"]["source_sha256"] = "changed"
    path.write_text(json.dumps({**body, "sha256": data.digest(body)}))
    with pytest.raises(ValueError, match="different source or grader"):
        data.load_manifest(path)
    path.write_text(json.dumps({**body, "sha256": "wrong"}))
    with pytest.raises(ValueError, match="integrity"):
        data.load_manifest(path)


def test_prepared_manifest_cannot_cross_model_response_format_policies(tmp_path):
    body = {
        "format": 1,
        "contract": data.data_contract("prompt", 128, 8, reasoning_required=False),
        "source_rows": 1,
        "included_rows": 1,
        "excluded_rows": 0,
        "records": [{"id": "question", "included": True}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({**body, "sha256": data.digest(body)}))
    with pytest.raises(ValueError, match="protocol disagree"):
        data.prepared_rows(tmp_path / "unread.parquet", path, "prompt", 8, 128, reasoning_required=True)


@pytest.mark.parametrize("model,required", [("Qwen/Qwen2.5-3B", False), ("Qwen/Qwen3-1.7B", True)])
def test_preparation_validates_pinned_model_response_format(study, monkeypatch, model, required):
    from deepseek_study.dataset import assets

    monkeypatch.setattr(assets, "MODEL_ID", model)
    assets.validate_grading_format(study.model_copy(update={"reasoning_required": required}))
    with pytest.raises(ValueError, match="pinned model contract"):
        assets.validate_grading_format(study.model_copy(update={"reasoning_required": not required}))


@pytest.mark.parametrize("changed", [False, True])
async def test_preparation_never_commits_results_with_a_changed_grader_contract(study, monkeypatch, changed):
    grader_identity = ["original"]
    requests = []
    closed = []
    row = {"id": "clock", "prompt": "What is the actual time on the clock?", "answers": ["6:00"]}

    class Pool:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            closed.append(True)

        async def call(self, request):
            requests.append(request)
            if changed:
                grader_identity[0] = "edited-while-grading"
            return {"status": "ok", "result": {"normalized_reference": "6:00"}}

    monkeypatch.setattr(data, "GraderPool", Pool)
    monkeypatch.setattr(data, "reward_identity", lambda: {"source_sha256": grader_identity[0]})
    monkeypatch.setattr(data, "read_rows", lambda path: [row])
    monkeypatch.setattr(data, "dataset_source", lambda path: data.source_for_sha256(data.DATASET_SHA256))
    monkeypatch.setattr(data, "tokenizer", lambda path: SimpleNamespace(apply_chat_template=lambda *args, **kw: [1]))
    if changed:
        with pytest.raises(ValueError, match="contract changed while grading references"):
            await data.prepare_data(study)
        assert not study.data_manifest.exists()
    else:
        result = await data.prepare_data(study)
        assert result["included_rows"] == 1
        manifest = data.load_manifest(study.data_manifest)
        assert manifest["contract"]["reward"]["source_sha256"] == "original"
    assert closed == [True]
    assert requests[0]["question"] == row["prompt"]


async def test_task_grading_uses_original_question_context_and_native_response_format(study, monkeypatch):
    import deepseek_deepscaler as environment
    from deepseek_study.dataset.grading import GraderPool

    question = "What is the actual time on the clock?"

    def prepared_rows(*args, **kwargs):
        assert kwargs["reasoning_required"] is False
        return [{"id": "clock", "prompt": question, "answers": ["6:00"]}]

    monkeypatch.setattr(environment, "prepared_rows", prepared_rows)
    monkeypatch.setattr(environment, "completion_tokens", lambda trace: [1])
    monkeypatch.setattr(
        environment, "tokenizer", lambda path: SimpleNamespace(decode=lambda *args, **kw: "\\boxed{6:00}")
    )
    config = environment.DeepScaleRConfig(
        dataset_path=str(study.dataset_path),
        data_manifest=str(study.data_manifest),
        tokenizer_path=str(study.prepared_model_path),
        prompt_instruction="Do not interpret this as a ratio.",
        truncated_reward="grade_final",
        reasoning_required=False,
        reward_timeout_seconds=5,
        reward_outer_timeout_seconds=10,
        reward_workers=1,
        reward_retries=0,
        seed=study.seed,
    )
    task = environment.DeepScaleRTaskset(config).load()[0]
    assert task.data.prompt != question
    assert task.data.reference_question == question
    trace = SimpleNamespace(info={}, is_truncated=False)
    async with GraderPool(workers=1, timeout=10, retries=0) as pool:
        monkeypatch.setattr(environment, "get_pool", lambda *args: pool)
        assert await task.correctness(trace) == 1
    assert trace.info["study_grading"]["result"]["reason"] == "correct"
