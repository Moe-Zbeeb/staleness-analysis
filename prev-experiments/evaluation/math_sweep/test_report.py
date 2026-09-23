import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sweep
from math_sweep.core import digest, file_hash, read_json, read_jsonl, seed_for, write_json


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = self.root / "prepared"
        self.output = self.root / "output"
        self.report = self.root / "report.json"
        self.rows = [
            {"benchmark": "aime24", "source_id": str(index), "prompt": f"Fixture question {index}",
             "answers": ["1"], "source_answers": ["001"], "heldout": index == 1,
             "overlap_status": "clear" if index == 1 else "confirmed_overlap",
             "prompt_token_ids": [11, index], "prompt_sha256": digest([11, index])}
            for index in (1, 2)
        ]
        self.base = self.make_cell("base", None, ((False, False), (True, True)))
        self.trained = self.make_cell("trained", 4, ((True, True), (True, True)))

    def make_cell(self, model_id, staleness, outcomes=None):
        model_dir = self.root / "models" / model_id
        model_dir.mkdir(parents=True)
        write_json(model_dir / "config.json", {"max_position_embeddings": 64, "eos_token_id": 151645})
        source_hashes = {str((model_dir / "config.json").resolve()): file_hash(model_dir / "config.json")}
        versions = {"verifiers": "0.3.1", "math-verify": "0.9.0", "sympy": "1.14.0"}
        cell = {
            "schema_version": 1, "availability": "pinned", "model": {"revision": "a" * 40},
            "model_id": model_id, "family": "qwen-test", "staleness": staleness,
            "checkpoint_step": 0 if staleness is None else 1000, "profile": "sampled-native",
            "profile_config": {"mode": "nonthinking", "context_kind": "native", "samples": 2,
                               "temperature": 0.6, "top_p": 1.0, "top_k": -1},
            "benchmark": "aime24", "budget": 8, "context": 64, "samples": 2, "seed": 42,
            "stop_token_ids": [151643, 151645], "expected_problems": 2, "expected_responses": 4,
            "comparison_sha256": "b" * 64, "input_files": source_hashes,
            "runtime": {"model_path": str(model_dir), "tokenizer_path": str(model_dir), "versions": versions},
        }
        cell["cell_id"] = digest(cell)[:24]
        prepared_dir = self.prepared / cell["cell_id"]
        prepared_dir.mkdir(parents=True)
        write_json(prepared_dir / "cell.json", cell)
        self.write_records(prepared_dir / "prompts.jsonl", self.rows)
        write_json(prepared_dir / "preparation.json", {
            "schema_version": 1,
            "source_cell_id": cell["cell_id"],
            "files": {name: file_hash(prepared_dir / name) for name in ("cell.json", "prompts.jsonl")},
            "input_files": source_hashes, "versions": versions,
        })
        if outcomes is None:
            return cell
        directory = self.output / cell["cell_id"]
        directory.mkdir(parents=True)
        preparation_sha = file_hash(prepared_dir / "preparation.json")
        records = []
        for sample in range(2):
            for index, row in enumerate(self.rows):
                correct = outcomes[index][sample]
                text = "Final answer: 1" if correct else "Final answer: 0"
                record = {key: cell[key] for key in ("cell_id", "model_id", "family", "staleness", "checkpoint_step", "profile", "benchmark", "budget", "comparison_sha256")}
                record.update(
                    source_id=row["source_id"], sample_index=sample,
                    seed=seed_for(cell["benchmark"], row["source_id"], sample, cell["seed"]),
                    prompt_sha256=row["prompt_sha256"], prompt_tokens=len(row["prompt_token_ids"]),
                    answers=row["answers"], source_answers=row["source_answers"], heldout=row["heldout"],
                    overlap_status=row["overlap_status"], mode="nonthinking", preparation_sha256=preparation_sha,
                    token_ids=[7 if correct else 8, 151645], token_count=2, engine_text=text,
                    raw_text=text + "<|im_end|>", text=text, finish_reason="stop", stop_reason=151645,
                    truncated=False, eos_seen=True, unfinished_thinking=False,
                    correct=correct, terminal_syntax=True, status="scored",
                )
                records.append(record)
        self.write_records(directory / "records.jsonl", records)
        allocation = {"job_id": "123", "host": "fixture-host", "cuda_visible_devices": ["0"], "visible_gpu_count": 1, "engine_gpu_count": 1}
        write_json(directory / "probes.json", {
            "schema_version": 1, "status": "passed", "cell_id": cell["cell_id"],
            "preparation_sha256": preparation_sha, "allocation": allocation,
            "gates": {
                "stop_151643": {"token_ids": [151643], "finish_reason": "stop", "stop_reason": 151643},
                "stop_151645": {"token_ids": [151645], "finish_reason": "stop", "stop_reason": 151645},
                "finite_normal": {"token_ids": [7], "selected_logprobs": [-0.5], "finish_reason": "stop", "stop_reason": 151645},
            },
        })
        write_json(directory / "receipt.json", {
            "schema_version": 1, "status": "complete", "cell_id": cell["cell_id"],
            "comparison_sha256": cell["comparison_sha256"], "preparation_sha256": preparation_sha,
            "records_sha256": file_hash(directory / "records.jsonl"),
            "probes_sha256": file_hash(directory / "probes.json"), "runtime_versions": versions,
            "expected_responses": 4, "responses": 4, "allocation": allocation,
            "elapsed_seconds": 60.0, "gpu_hours": 1 / 60, "timing_complete": True,
            "attempts": [{"index": 1, "status": "complete", "elapsed_seconds": 60.0, "gpu_hours": 1 / 60, "timing_complete": True, "allocation": allocation}],
        })
        return cell

    @staticmethod
    def write_records(path, records):
        Path(path).write_text("".join(json.dumps(record) + "\n" for record in records))

    def cli(self, allow_partial=False):
        argv = ["sweep.py", "report", "--prepared-root", str(self.prepared), "--output-root", str(self.output),
                "--report", str(self.report), "--bootstrap-samples", "100"]
        if allow_partial:
            argv.append("--allow-partial")
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            sweep.main()
        return read_json(self.report)

    def refresh_record_hash(self, cell):
        directory = self.output / cell["cell_id"]
        receipt = read_json(directory / "receipt.json")
        receipt["records_sha256"] = file_hash(directory / "records.jsonl")
        write_json(directory / "receipt.json", receipt)

    def test_cli_complete_pair_and_filtered_membership(self):
        result = self.cli()
        self.assertEqual(result["status"], "complete_for_prepared_cells")
        self.assertEqual(result["full"]["responses"], 8)
        self.assertEqual(result["training_overlap_filtered"]["responses"], 4)
        self.assertEqual(result["full"]["paired_deltas"][0]["delta_pp"], 50)
        self.assertEqual(result["training_overlap_filtered"]["paired_deltas"][0]["delta_pp"], 100)
        self.assertTrue(all(cell["problems"] == 2 for cell in result["full"]["cells"]))
        self.assertTrue(all(cell["problems"] == 1 for cell in result["training_overlap_filtered"]["cells"]))
        self.assertEqual(result["runtime"]["elapsed_seconds"], 120)
        self.assertAlmostEqual(result["runtime"]["gpu_hours"], 1 / 30)
        self.assertEqual(len(result["provenance"]), 2)
        markdown = self.report.with_suffix(".md").read_text()
        self.assertIn("Full benchmark membership", markdown)
        self.assertIn("Confirmed training overlaps excluded", markdown)

    def test_partial_prepared_scope_requires_explicit_flag(self):
        missing = self.make_cell("pending", 6)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.cli()
        self.assertFalse(self.report.exists())
        result = self.cli(allow_partial=True)
        self.assertEqual(result["status"], "partial")
        self.assertEqual([cell["cell_id"] for cell in result["missing_cells"]], [missing["cell_id"]])
        self.assertEqual(result["full"]["responses"], 8)
        self.assertIn("Status: partial", self.report.with_suffix(".md").read_text())

    def test_changed_record_bytes_and_completion_hash_are_rejected(self):
        directory = self.output / self.trained["cell_id"]
        records_path = directory / "records.jsonl"
        original = records_path.read_bytes()
        records_path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "completion receipt"):
            self.cli()
        records_path.write_bytes(original)
        receipt = read_json(directory / "receipt.json")
        receipt["records_sha256"] = "0" * 64
        write_json(directory / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "completion receipt"):
            self.cli()

    def test_changed_probe_bytes_are_rejected(self):
        path = self.output / self.trained["cell_id"] / "probes.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "probe receipt"):
            self.cli()

    def test_preparation_binding_is_required(self):
        directory = self.output / self.trained["cell_id"]
        receipt = read_json(directory / "receipt.json")
        receipt["preparation_sha256"] = "wrong"
        write_json(directory / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "different preparation"):
            self.cli()

    def test_prepared_prompt_hash_is_required(self):
        path = self.prepared / self.trained["cell_id"] / "prompts.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Prepared inputs changed"):
            self.cli()

    def test_record_cohort_cannot_be_changed_even_with_updated_file_hash(self):
        directory = self.output / self.trained["cell_id"]
        path = directory / "records.jsonl"
        records = read_jsonl(path)
        records[0]["heldout"] = False
        self.write_records(path, records)
        self.refresh_record_hash(self.trained)
        with self.assertRaisesRegex(ValueError, "membership mismatch"):
            self.cli()

    def test_incomplete_records_cannot_claim_completion(self):
        directory = self.output / self.trained["cell_id"]
        path = directory / "records.jsonl"
        self.write_records(path, read_jsonl(path)[:-1])
        self.refresh_record_hash(self.trained)
        with self.assertRaises(ValueError):
            self.cli()

    def test_receipt_identity_and_counts_must_match(self):
        path = self.output / self.trained["cell_id"] / "receipt.json"
        original = path.read_bytes()
        changes = {"cell_id": "another-cell", "comparison_sha256": "wrong-comparison", "expected_responses": 3, "responses": 3}
        for name, value in changes.items():
            with self.subTest(field=name):
                receipt = json.loads(original)
                receipt[name] = value
                write_json(path, receipt)
                with self.assertRaises(ValueError):
                    self.cli()
                path.write_bytes(original)

    def test_hashed_probes_must_be_passed_for_this_preparation(self):
        directory = self.output / self.trained["cell_id"]
        path = directory / "probes.json"
        receipt_path = directory / "receipt.json"
        original_probe, original_receipt = path.read_bytes(), receipt_path.read_bytes()
        for name, value in (("status", "failed"), ("cell_id", "another-cell"), ("preparation_sha256", "another-preparation")):
            with self.subTest(field=name):
                probe = json.loads(original_probe)
                probe[name] = value
                write_json(path, probe)
                receipt = json.loads(original_receipt)
                receipt["probes_sha256"] = file_hash(path)
                write_json(receipt_path, receipt)
                with self.assertRaises(ValueError):
                    self.cli()
                path.write_bytes(original_probe)
                receipt_path.write_bytes(original_receipt)


if __name__ == "__main__":
    unittest.main()
