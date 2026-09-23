import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from . import core, data


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.retained = [{"benchmark": "aime24", "source_id": "a1", "prompt": "Calculate the number of green tiles in the described square.", "answers": ["042"]}]
        self.training = [{"benchmark": "dapo_train", "source_id": "train1", "prompt": "Find all prime factors of the positive integer 143.", "answers": ["11,13"]}]
        self.added = [{"problem_idx": 10, "problem": "Determine all complex solutions of the polynomial shown here.", "answer": "2,3,\\frac{3+i\\sqrt{3}}{2},\\frac{3-i\\sqrt{3}}{2}"}]
        self.catalog = {"schema_version": 1, "benchmarks": [{"id": "aime24", "repo_id": "test/retained", "revision": "a" * 40, "split": "train", "expected_count": 1, "source": "retained", "answer_kind": "aime_integer", "fields": {"prompt": "prompt", "answers": "answers", "source_id": "source_id"}}, {"id": "brumo_2025", "repo_id": "test/added", "revision": "b" * 40, "split": "train", "expected_count": 1, "source": "huggingface", "answer_kind": "math", "fields": {"prompt": "problem", "answers": "answer", "source_id": "problem_idx"}}]}
        self.write_inputs()

    def write_inputs(self):
        for label, records, filename in [("retained_source", self.retained, "eval.jsonl"), ("training_source", self.training, "train.jsonl")]:
            payload = "".join(json.dumps(record) + "\n" for record in records).encode()
            (self.root / filename).write_bytes(payload)
            self.catalog[label] = {"path": filename, "sha256": hashlib.sha256(payload).hexdigest(), "count": len(records)}

    def prepare(self, **kwargs):
        with patch.object(data, "_download_rows", return_value=self.added):
            return data.prepare_data(self.catalog, self.root, **kwargs)

    def test_raw_gold_and_complete_root_lists_are_preserved(self):
        result = self.prepare()
        self.assertEqual(result["rows"][0]["answers"], ["42"])
        self.assertEqual(result["rows"][0]["source_answers"], ["042"])
        self.assertEqual(result["rows"][1]["answers"], [self.added[0]["answer"]])
        self.assertEqual(result["overlap_audit"]["heldout_count"], 2)

    def test_aime_normalization_matches_existing_policy(self):
        for answer in ["042", "42°", "42^\\circ", "42^{\\circ}"]:
            self.assertEqual(data.normalize_aime(answer), "42")
        for answer in ["42.0", "-1", "1000", "42,43", "x=42"]:
            with self.assertRaises(ValueError):
                data.normalize_aime(answer)

    def test_missing_or_changed_frozen_file_fails(self):
        (self.root / "eval.jsonl").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "do not reconstruct"):
            self.prepare()
        self.write_inputs()
        (self.root / "train.jsonl").write_text("[]\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.prepare()

    def test_wrong_count_empty_answer_and_duplicate_are_rejected(self):
        self.catalog["benchmarks"][1]["expected_count"] = 2
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            self.prepare()
        self.catalog["benchmarks"][1]["expected_count"] = 1
        self.added[0]["answer"] = ""
        with self.assertRaisesRegex(ValueError, "Empty gold"):
            self.prepare()
        self.added[0]["answer"] = "3"
        self.added.append(dict(self.added[0]))
        self.catalog["benchmarks"][1]["expected_count"] = 2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.prepare()

    def test_exact_and_normalized_overlap_and_instruction_stripping(self):
        self.training[0]["prompt"] = core.INSTRUCTION + self.retained[0]["prompt"]
        self.write_inputs()
        result = self.prepare()
        self.assertEqual(result["rows"][0]["overlap_status"], "excluded_exact")
        self.assertFalse(result["rows"][0]["heldout"])
        self.training[0]["prompt"] = self.retained[0]["prompt"].replace("green", "\\mathrm{green}")
        self.write_inputs()
        self.assertEqual(self.prepare()["rows"][0]["overlap_status"], "excluded_normalized")

    def test_near_overlap_requires_bound_review(self):
        self.training[0]["prompt"] = self.retained[0]["prompt"].replace("green", "brown")
        self.write_inputs()
        result = self.prepare()
        row = result["rows"][0]
        self.assertEqual(row["overlap_status"], "unresolved_near_duplicate")
        self.assertFalse(row["heldout"])
        self.assertEqual(result["overlap_audit"]["rows"][0]["matches"][0]["training_source_id"], "train1")
        decision = {row["row_key"]: {"decision": "keep", "evidence": "The two wordings describe different mathematical inputs.", "prompt_sha256": row["prompt_sha256"]}}
        reviewed = self.prepare(decisions=decision)
        self.assertEqual(reviewed["rows"][0]["overlap_status"], "reviewed_keep")
        self.assertTrue(reviewed["rows"][0]["heldout"])
        decision[row["row_key"]]["prompt_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "prompt hash mismatch"):
            self.prepare(decisions=decision)

    def test_confirmed_overlap_cannot_be_retained(self):
        self.training[0]["prompt"] = self.retained[0]["prompt"]
        self.write_inputs()
        row = self.prepare()["rows"][0]
        with self.assertRaisesRegex(ValueError, "Cannot retain"):
            self.prepare(decisions={row["row_key"]: {"decision": "keep", "evidence": "Ignore match", "prompt_sha256": row["prompt_sha256"]}})

    def test_small_numeric_variants_are_reviewed(self):
        self.training[0]["prompt"] = "Find the value of 12345 plus 54321."
        self.retained[0]["prompt"] = "Find the value of 12346 plus 54321."
        self.write_inputs()
        self.assertEqual(self.prepare()["rows"][0]["overlap_status"], "unresolved_near_duplicate")

    def test_chinese_training_questions_keep_auditable_content(self):
        self.training[0]["prompt"] = "单位圆的内接五边形的所有边及所有对角线的长度的平方和的最大值为多少？"
        self.write_inputs()
        self.assertTrue(data.normalized(self.training[0]["prompt"]))
        self.assertEqual(self.prepare()["overlap_audit"]["training_count"], 1)

    def test_cross_benchmark_duplicates_are_audited(self):
        self.added[0]["problem"] = self.retained[0]["prompt"]
        result = self.prepare()
        self.assertEqual(result["overlap_audit"]["cross_benchmark_duplicates"], [["aime24:a1", "brumo_2025:10"]])

    def test_pinned_download_checks_bytes_before_reading(self):
        path = self.root / "file.parquet"
        path.write_bytes(b"test parquet bytes")
        spec = dict(self.catalog["benchmarks"][1], files=["data/train.parquet"], file_sha256={"data/train.parquet": core.file_hash(path)})
        hub = types.ModuleType("huggingface_hub")
        arrow = types.ModuleType("pyarrow")
        parquet = types.ModuleType("pyarrow.parquet")
        hub.hf_hub_download = lambda **kwargs: str(path)
        parquet.read_table = lambda p: types.SimpleNamespace(to_pylist=lambda: self.added)
        arrow.parquet = parquet
        with patch.dict("sys.modules", {"huggingface_hub": hub, "pyarrow": arrow, "pyarrow.parquet": parquet}):
            self.assertEqual(data._download_rows(spec), self.added)
            path.write_bytes(b"changed parquet")
            with self.assertRaisesRegex(ValueError, "parquet SHA256 mismatch"):
                data._download_rows(spec)
            spec["revision"] = "main"
            with self.assertRaisesRegex(ValueError, "Unpinned dataset"):
                data._download_rows(spec)

    def test_decisions_require_evidence_and_known_rows(self):
        with self.assertRaisesRegex(ValueError, "Unknown overlap decision"):
            self.prepare(decisions={"absent:99": {"decision": "keep"}})
        with self.assertRaisesRegex(ValueError, "requires keep/exclude and evidence"):
            self.prepare(decisions={"aime24:a1": "exclude"})


if __name__ == "__main__":
    unittest.main()
