import json
from pathlib import Path

import pytest

from deepseek_study import data
from deepseek_study.recipe import baseline


def test_prepared_full_dataset_has_explicit_accounted_exclusions():
    root = Path(__file__).resolve().parents[1]
    path = root / "assets/train-manifest.json"
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
