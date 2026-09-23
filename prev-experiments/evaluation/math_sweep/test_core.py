import copy
import tempfile
import unittest
from pathlib import Path

from math_sweep.core import check_context, digest, load_catalogs, make_plan, seed_for, validate_records
from math_sweep.prepare import tokenize_rows, verify_model


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.catalogs = load_catalogs()
        self.plan = make_plan(*self.catalogs)

    def test_matrix_and_no_duplicate_conditions(self):
        self.assertEqual(self.plan["summary"]["models"], 20)
        self.assertEqual(self.plan["summary"]["cells"], 560)
        self.assertEqual(self.plan["summary"]["responses_before_overlap_filter"], 194480)
        self.assertEqual(len({cell["cell_id"] for cell in self.plan["cells"]}), 560)
        self.assertEqual(len([item for item in self.catalogs[0]["models"] if item["status"] == "pinned"]), 13)

    def test_pending_models_have_no_invented_revision(self):
        for model in self.catalogs[0]["models"]:
            if model["status"] == "awaiting_final":
                self.assertIsNone(model["revision"])

    def test_invalid_selection_fails(self):
        with self.assertRaises(ValueError):
            make_plan(*self.catalogs, selected_models=["qwen-fake"])

    def test_experiments_are_opt_in(self):
        self.assertTrue(all(cell["profile_config"]["context_kind"] == "native" for cell in self.plan["cells"]))
        extension = make_plan(*self.catalogs, selected_profiles=["15b-extension-sampled"])
        self.assertEqual({cell["context"] for cell in extension["cells"]}, {9216})
        self.assertEqual({cell["budget"] for cell in extension["cells"]}, {3072, 8192})

    def test_context_rejects_overflow_without_mutation(self):
        ids = [1] * 1025
        with self.assertRaises(ValueError):
            check_context(ids, 3072, 4096)
        self.assertEqual(len(ids), 1025)
        check_context([1] * 1024, 3072, 4096)

    def test_seed_stable_across_model_and_budget(self):
        self.assertEqual(seed_for("aime24", "1", 0), seed_for("aime24", "1", 0))
        self.assertNotEqual(seed_for("aime24", "1", 0), seed_for("aime24", "1", 1))
        self.assertNotEqual(seed_for("aime24", "1", 0), seed_for("aime25", "1", 0))

    def test_resume_integrity(self):
        cell = {"cell_id": "test", "samples": 2, "benchmark": "aime24", "seed": 42, "budget": 3072,
                "model_id": "trained", "family": "qwen-test", "profile": "sampled", "staleness": 4,
                "comparison_sha256": "comparison", "checkpoint_step": 1000}
        rows = [{"source_id": "1", "prompt_sha256": "prompt", "answers": ["1"], "heldout": True}]
        record = {**cell, "source_id": "1", "sample_index": 0, "prompt_sha256": "prompt", "answers": ["1"],
                  "seed": seed_for("aime24", "1", 0), "token_ids": [1, 2], "token_count": 2, "finish_reason": "stop",
                  "heldout": True, "truncated": False}
        self.assertEqual(validate_records([record], rows, cell), {("1", 0)})
        with self.assertRaises(ValueError):
            validate_records([record], rows, cell, complete=True)
        with self.assertRaises(ValueError):
            validate_records([record, record], rows, cell)
        for field, bad in (("answers", ["2"]), ("prompt_sha256", "changed"), ("seed", 1), ("cell_id", "other"), ("token_count", 1)):
            with self.assertRaises(ValueError):
                validate_records([{**record, field: bad}], rows, cell)

    def test_export_rejects_wrong_step(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text('{"max_position_embeddings":4096}')
            (path / "model.safetensors").write_bytes(b"fixture")
            (path / "export-manifest.json").write_text('{"status":"passed","step":999}')
            with self.assertRaisesRegex(ValueError, "step-1000"):
                verify_model(path, {"staleness": 2}, {})

    def test_mode_and_overlap_gates(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return [1]
            def decode(self, *args, **kwargs):
                return "<think>open"
        cell = copy.deepcopy(next(cell for cell in self.plan["cells"] if cell["family"] == "qwen3-14b"))
        cell["expected_problems"] = 1
        row = {"benchmark": cell["benchmark"], "source_id": "1", "prompt": "p", "answers": ["1"], "heldout": True}
        with self.assertRaisesRegex(ValueError, "mode mismatch"):
            tokenize_rows([row], Tokenizer(), cell)
        with self.assertRaisesRegex(ValueError, "Review training overlap"):
            tokenize_rows([{**row, "overlap_status": "unresolved_near_duplicate"}], Tokenizer(), cell)


if __name__ == "__main__":
    unittest.main()
