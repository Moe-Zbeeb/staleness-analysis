import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from deepseek_study.dataset import assets, prepare, sources


@pytest.fixture
def dapo_file(tmp_path, monkeypatch):
    rows = [
        {
            "source_id": "original-question-a",
            "prompt": "Compute 1 + 1.",
            "answers": ["2"],
            "prompt_tokens": 3,
            "benchmark": "dapo_train",
        },
        {
            "source_id": "original-question-b",
            "prompt": "Compute 2 + 2.",
            "answers": ["4"],
            "prompt_tokens": 3,
            "benchmark": "dapo_train",
        },
    ]
    path = tmp_path / "train.jsonl"

    def write(records):
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        source = replace(sources.DAPO_17K, rows=len(records), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        monkeypatch.setitem(sources.SOURCES, source.sha256, source)
        return source

    source = write(rows)
    return path, rows, source, write


def test_dapo_adapter_preserves_question_identity_text_answers_and_order(dapo_file):
    path, original, source, _ = dapo_file
    rows = assets.read_rows(path)
    assert assets.dataset_source(path) == source
    for raw, row in zip(original, rows, strict=True):
        assert row["id"] == row["source_id"] == raw["source_id"]
        assert row["prompt"] == raw["prompt"]
        assert row["answers"] == raw["answers"]
        assert row["benchmark"] == raw["benchmark"]
        assert row["messages"] == [{"role": "user", "content": raw["prompt"]}]
        assert row["source_prompt_tokens"] == raw["prompt_tokens"]
        assert "prompt_tokens" not in row


def test_unknown_dataset_and_manifest_cross_source_are_rejected(dapo_file, tmp_path):
    path, _, _, _ = dapo_file
    with pytest.raises(ValueError, match="different locked training releases"):
        assets.read_rows(path, expected_sha256=sources.DEEPSCALER.sha256)
    unknown = tmp_path / "unknown.jsonl"
    unknown.write_text("{}\n")
    with pytest.raises(ValueError, match="every locked training release"):
        assets.read_rows(unknown)


@pytest.mark.parametrize("invalid", ["duplicate", "empty_prompt", "multiple_answers", "empty_answer"])
def test_dataset_adapter_rejects_ambiguous_or_invalid_questions(dapo_file, invalid):
    path, rows, _, write = dapo_file
    if invalid == "duplicate":
        rows[1]["source_id"] = rows[0]["source_id"]
    elif invalid == "empty_prompt":
        rows[0]["prompt"] = " "
    elif invalid == "multiple_answers":
        rows[0]["answers"] = ["2", "two"]
    else:
        rows[0]["answers"] = [""]
    write(rows)
    with pytest.raises(ValueError):
        assets.read_rows(path)


def test_legacy_contract_is_unchanged_and_new_manifest_pins_source_metadata(dapo_file, tmp_path):
    _, _, source, _ = dapo_file
    legacy = prepare.data_contract("instruction", 2048, 8)
    assert legacy["source_sha256"] == sources.DEEPSCALER.sha256
    assert "source" not in legacy
    contract = prepare.data_contract("instruction", 2048, 8, source_sha256=source.sha256)
    assert contract["source"] == source.identity()
    body = {"format": 1, "contract": contract}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({**body, "sha256": prepare.digest(body)}))
    assert prepare.load_manifest(manifest)["contract"] == contract
    contract["source"]["revision"] = "unlocked"
    manifest.write_text(json.dumps({**body, "sha256": prepare.digest(body)}))
    with pytest.raises(ValueError, match="source metadata"):
        prepare.load_manifest(manifest)


async def test_new_source_rechecks_prompt_tokens_and_references_with_current_protocol(
    dapo_file, study, monkeypatch
):
    path, rows, source, _ = dapo_file
    calls = []
    messages = []

    class Pool:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def call(self, request):
            calls.append(request)
            return {"status": "ok", "result": {"normalized_reference": request["answer"]}}

    def tokenize(prompt, **kwargs):
        messages.append(prompt)
        return list(range(10 if rows[0]["prompt"] in prompt[0]["content"] else 200))

    monkeypatch.setattr(prepare, "GraderPool", Pool)
    monkeypatch.setattr(prepare, "tokenizer", lambda path: SimpleNamespace(apply_chat_template=tokenize))
    study = study.model_copy(update={"dataset_path": path, "prompt_instruction": "Show your work."})
    summary = await prepare.prepare_data(study)
    assert summary["source_rows"] == 2
    assert summary["included_rows"] == 1
    assert summary["excluded_rows"] == 1
    manifest = prepare.load_manifest(study.data_manifest)
    assert manifest["contract"]["source"] == source.identity()
    assert [row["prompt_tokens"] for row in manifest["records"]] == [10, 200]
    assert manifest["records"][1]["reason"] == "prompt_over_limit"
    assert len(calls) == 1
    assert calls[0]["question"] == rows[0]["prompt"]
    assert calls[0]["answer"] == "2"
    assert all(message[0]["content"].endswith("\n\nShow your work.") for message in messages)
    retained = prepare.prepared_rows(
        path, study.data_manifest, study.prompt_instruction, study.reward_timeout_seconds, study.prompt_max_tokens
    )
    assert [row["id"] for row in retained] == [rows[0]["source_id"]]


def test_pinned_public_dapo_release_when_fixture_available():
    path = Path("/tmp/staleness-dapo-17k-53064564/train.jsonl")
    if not path.is_file():
        pytest.skip("Pinned public DAPO fixture has not been downloaded")
    rows = assets.read_rows(path)
    assert len(rows) == 17005
    assert len({row["id"] for row in rows}) == 17005
    assert all(row["id"] == row["source_id"] for row in rows)
