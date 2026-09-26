from pathlib import Path

import pytest

from deepseek_study.dataset.assets import prepare_tokenizer, read_rows


def test_native_tokenizer_and_official_renderer_match(tmp_path):
    fixture = Path(__file__).resolve().parents[1] / "assets" / "tokenizer-fixture"
    if not (fixture / "tokenizer.json").is_file():
        pytest.skip("Pinned tokenizer fixture has not been downloaded")
    result = prepare_tokenizer(fixture, tmp_path / "prepared")
    assert result["probes_passed"] == 5
    assert result["native_template"]


def test_cleaned_release_identity_and_answer_shape():
    fixture = Path(__file__).resolve().parents[1] / "assets" / "dataset-fixture" / "data" / "train.parquet"
    if not fixture.is_file():
        pytest.skip("Pinned dataset fixture has not been downloaded")
    rows = read_rows(fixture)
    assert len(rows) == 37713
