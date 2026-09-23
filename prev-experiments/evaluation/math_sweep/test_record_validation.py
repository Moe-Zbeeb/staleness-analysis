import unittest

from math_sweep.core import seed_for, validate_records


class RecordValidationTests(unittest.TestCase):
    def setUp(self):
        self.cell = {
            "cell_id": "cell", "model_id": "trained", "family": "qwen-test", "profile": "sampled-native",
            "benchmark": "aime24", "staleness": 4, "budget": 3072, "comparison_sha256": "comparison",
            "checkpoint_step": 1000, "seed": 42, "samples": 1,
        }
        self.row = {"source_id": "1", "prompt_sha256": "prompt", "answers": ["42"], "heldout": True}
        self.record = {
            **self.cell, **self.row, "sample_index": 0, "seed": seed_for("aime24", "1", 0),
            "token_count": 2, "token_ids": [12, 151645], "finish_reason": "stop", "truncated": False,
        }

    def test_valid_complete_cell(self):
        self.assertEqual(validate_records([self.record], [self.row], self.cell, complete=True), {("1", 0)})

    def test_result_labels_are_bound_to_prepared_cell(self):
        changes = {
            "cell_id": "other-cell", "model_id": "base", "family": "other-family", "profile": "greedy-native",
            "benchmark": "aime25", "staleness": 8, "budget": 8192, "comparison_sha256": "other-comparison",
            "checkpoint_step": 999,
        }
        for key, value in changes.items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity mismatch"):
                validate_records([{**self.record, key: value}], [self.row], self.cell, complete=True)

    def test_missing_identity_cannot_silently_pass(self):
        for key in ("model_id", "family", "profile", "staleness", "budget", "comparison_sha256", "checkpoint_step"):
            record = {name: value for name, value in self.record.items() if name != key}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity mismatch"):
                validate_records([record], [self.row], self.cell)

    def test_heldout_flag_must_match_prepared_question(self):
        for value in (False, 1, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "membership mismatch"):
                validate_records([{**self.record, "heldout": value}], [self.row], self.cell)

    def test_truncation_must_match_stop_reason(self):
        for change in ({"truncated": True}, {"finish_reason": "length"}, {"truncated": 0}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "Truncation"):
                validate_records([{**self.record, **change}], [self.row], self.cell)
        self.assertTrue(validate_records([{**self.record, "finish_reason": "length", "truncated": True}], [self.row], self.cell))

    def test_boolean_indices_and_invalid_tokens_are_rejected(self):
        for change in ({"sample_index": False}, {"token_count": True}, {"token_ids": [True, 151645]}, {"token_ids": [-1, 151645]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_records([{**self.record, **change}], [self.row], self.cell)

    def test_duplicate_prepared_questions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate prepared"):
            validate_records([self.record], [self.row, self.row], self.cell)


if __name__ == "__main__":
    unittest.main()
