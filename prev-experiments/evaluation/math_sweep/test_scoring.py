import copy
import importlib.util
import math
import unittest

from math_sweep import scoring


class Tokenizer:
    vocabulary = {1: "<think>", 2: "reasoning ", 3: "</think>", 4: "Final answer: 42", 5: "<|im_end|>", 6: "<|endoftext|>", 7: " incorrect tail"}

    def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        if skip_special_tokens or clean_up_tokenization_spaces:
            raise AssertionError("Thinking tags must survive decoding")
        return "".join(self.vocabulary[token] for token in ids)


def records(model_id="base", staleness=None, outcomes=((False, False, True, True), (False, False, False, False))):
    return [{
        "model_id": model_id,
        "family": "qwen-test",
        "staleness": staleness,
        "profile": "native-sampled",
        "benchmark": "aime24",
        "budget": 3072,
        "source_id": str(question),
        "sample_index": sample,
        "correct": correct,
        "token_count": 100 + sample,
        "truncated": False,
        "terminal_syntax": True,
        "prompt_sha256": f"prompt-{question}",
        "seed": 42 + question * 100 + sample,
        "answers": ["42"],
    } for question, answers in enumerate(outcomes) for sample, correct in enumerate(answers)]


class DecodeTests(unittest.TestCase):
    def test_closed_reasoning_and_terminal_eos(self):
        result = scoring.decode_completion([1, 2, 3, 4, 5], Tokenizer(), [5, 6])
        self.assertEqual(result["text"], "Final answer: 42")
        self.assertIn("<think>", result["raw_text"])
        self.assertTrue(result["eos_seen"])
        self.assertFalse(result["unfinished_thinking"])

    def test_prompt_open_thinking_without_closing_token(self):
        decoded = scoring.decode_completion([2, 4, 6], Tokenizer(), [5, 6], thinking_open=True)
        self.assertTrue(decoded["unfinished_thinking"])
        self.assertEqual(scoring.grade(decoded["text"], ["42"], decoded["unfinished_thinking"])["status"], "unfinished_thinking")

    def test_prompt_open_thinking_can_close(self):
        decoded = scoring.decode_completion([2, 3, 4], Tokenizer(), [5, 6], thinking_open=True)
        self.assertFalse(decoded["unfinished_thinking"])
        self.assertFalse(decoded["eos_seen"])
        self.assertEqual(decoded["text"], "Final answer: 42")

    def test_post_eos_tokens_are_rejected(self):
        for ids in ([4, 5, 7], [4, 6, 5], [5, 5]):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "after EOS"):
                scoring.decode_completion(ids, Tokenizer(), [5, 6])

    def test_reopened_thinking_is_unfinished(self):
        decoded = scoring.decode_completion([1, 2, 3, 4, 1], Tokenizer(), [5, 6])
        self.assertTrue(decoded["unfinished_thinking"])


class SyntaxTests(unittest.TestCase):
    def test_terminal_nested_fraction_and_trailing_prose(self):
        self.assertTrue(scoring._terminal_syntax(r"Therefore, \boxed{\frac{1}{2}}."))
        self.assertFalse(scoring._terminal_syntax("Final answer: 42\nStill working"))
        self.assertFalse(scoring._terminal_syntax(r"\boxed{\frac{1}{2}"))
        self.assertTrue(scoring._terminal_syntax("\\[\n\\boxed{42}\n\\]"))

    def test_unfinished_thinking_is_not_graded(self):
        self.assertEqual(scoring.grade("<think>Final answer: 42", ["42"])["status"], "unfinished_thinking")
        self.assertEqual(scoring.grade("Maybe 42", ["42"])["status"], "no_terminal_answer")


@unittest.skipUnless(importlib.util.find_spec("verifiers"), "Frozen verifier runtime is not installed")
class FrozenGraderTests(unittest.TestCase):
    def test_matharena_runtime_fixtures(self):
        validation = scoring.validate_grader()
        self.assertTrue(validation["success"])
        self.assertEqual(len(validation["cases"]), 10)
        self.assertEqual(set(validation["versions"]), {"verifiers", "math-verify", "sympy", "latex2sympy2-extended"})

    def test_strict_final_answer_semantics(self):
        cases = [
            ("Final answer: 042", ["42"], True),
            (r"\boxed{\frac{1}{2}}", ["0.5"], True),
            ("Final answer: 42\nFinal answer: 41", ["42"], False),
            ("Final answer: 42\nUnfinished reasoning", ["42"], False),
            (r"\boxed{42", ["42"], False),
            (r"\boxed{}", ["42"], False),
        ]
        for text, answers, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(scoring.grade(text, answers)["correct"], expected)
        self.assertEqual(scoring.grader_provenance()["sha256"], scoring.GRADER_SHA256)


class SummaryTests(unittest.TestCase):
    def test_pass_at_k_exact_oracle_estimator(self):
        self.assertAlmostEqual(scoring.pass_at_k(16, 1, 4), 0.25)
        self.assertAlmostEqual(scoring.pass_at_k(16, 2, 4), 1 - math.comb(14, 4) / math.comb(16, 4))
        self.assertEqual(scoring.pass_at_k(16, 1, 16), 1)
        self.assertEqual(scoring.pass_at_k(16, 0, 16), 0)
        with self.assertRaises(ValueError):
            scoring.pass_at_k(8, 1, 16)

    def test_balanced_problem_statistics_and_matching(self):
        data = records() + records("stale4", 4, ((True, True, True, True), (False, False, True, True)))
        summary = scoring.summarize(data, bootstrap_samples=200, seed=7)
        by_model = {cell["model_id"]: cell for cell in summary["cells"]}
        self.assertEqual(by_model["base"]["mean_accuracy"], 0.25)
        self.assertEqual(by_model["base"]["pass_at_k"], {"1": 0.25, "4": 0.5})
        self.assertEqual(summary["paired_deltas"][0]["delta_pp"], 50)
        self.assertEqual(summary["paired_deltas"][0]["delta_ci95_pp"], [50, 50])
        self.assertEqual(summary, scoring.summarize(reversed(data), bootstrap_samples=200, seed=7))

    def test_bootstrap_resamples_questions_not_completions(self):
        data = records(outcomes=((True,) * 16, (False,) * 16))
        summary = scoring.summarize(data, bootstrap_samples=2000, seed=42)
        self.assertEqual(summary["cells"][0]["accuracy_ci95"], [0, 1])

    def test_duplicate_or_missing_samples_fail(self):
        data = records()
        for changed in (data + [data[0]], data[:-1], [row for row in data if row["sample_index"] != 0]):
            with self.subTest(rows=len(changed)), self.assertRaises(ValueError):
                scoring.summarize(changed, bootstrap_samples=0)

    def test_pairing_rejects_different_seeds(self):
        trained = records("stale6", 6)
        trained[0]["seed"] += 1
        summary = scoring.summarize(records() + trained, bootstrap_samples=0)
        self.assertFalse(summary["paired_deltas"])
        self.assertEqual(summary["unpaired"][0]["reason"], "paired_seed_mismatch")

    def test_pairing_rejects_different_comparison_profiles_and_cohorts(self):
        for field, base_value, trained_value in (("comparison_sha256", "engine-a", "engine-b"), ("heldout", True, False)):
            base, trained = records(), records("stale4", 4)
            for row in base:
                row[field] = base_value
            for row in trained:
                row[field] = trained_value
            summary = scoring.summarize(base + trained, bootstrap_samples=0)
            self.assertEqual(summary["unpaired"][0]["reason"], f"paired_{field}_mismatch")

    def test_problem_identity_cannot_change_between_samples(self):
        data = records()
        data[1]["prompt_sha256"] = "different-question"
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            scoring.summarize(data, bootstrap_samples=0)

    def test_pairing_does_not_silently_intersect_questions(self):
        trained = records("stale8", 8)
        for row in trained:
            if row["source_id"] == "1":
                row["source_id"] = "other"
        summary = scoring.summarize(records() + trained, bootstrap_samples=0)
        self.assertEqual(summary["unpaired"][0]["reason"], "question_or_sample_mismatch")

    def test_profiles_and_budgets_remain_separate(self):
        base = records()
        other = copy.deepcopy(base)
        for row in other:
            row["profile"] = "experimental-extension-sampled"
            row["budget"] = 8192
        summary = scoring.summarize(base + other, bootstrap_samples=0)
        self.assertEqual(len(summary["cells"]), 2)
        self.assertIn("8192", scoring.render_markdown(summary))

    def test_invalid_measurements_fail(self):
        for key, value in (("correct", 1), ("token_count", 4000), ("staleness", "base")):
            data = records()
            data[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                scoring.summarize(data, bootstrap_samples=0)


if __name__ == "__main__":
    unittest.main()
