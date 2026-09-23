import fcntl
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from math_sweep import runner
from math_sweep.core import digest, file_hash, read_json, read_jsonl, write_json


class Tokenizer:
    def decode(self, ids, **kwargs):
        return "Final answer: 1" if ids else ""


class FakeEngine:
    def __init__(self, fail_batch=None, wrong_eos=False, nonfinite=False):
        self.batches = []
        self.fail_batch = fail_batch
        self.wrong_eos = wrong_eos
        self.nonfinite = nonfinite

    def generate(self, prompts, params, use_tqdm=False):
        batch = isinstance(params, list)
        if batch:
            self.batches.append(params)
            if len(self.batches) == self.fail_batch:
                raise RuntimeError("injected generation failure")
        else:
            params = [params] * len(prompts)
        result = []
        for prompt, setting in zip(prompts, params):
            if hasattr(setting, "logit_bias"):
                eos = next(iter(setting.logit_bias))
                tokens = [999] if self.wrong_eos else [eos]
                stop = eos
                reason = "stop"
            else:
                tokens, stop, reason = [7], None, "length"
            probabilities = [{token: SimpleNamespace(logprob=float("nan") if self.nonfinite else -0.5)} for token in tokens]
            reply = SimpleNamespace(token_ids=tokens, text="Final answer: 1", finish_reason=reason, stop_reason=stop, logprobs=probabilities)
            result.append(SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"], outputs=[reply], finished=True))
        return result


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        self.model = self.root / "model"
        self.model.mkdir()
        write_json(self.model / "config.json", {"max_position_embeddings": 64, "eos_token_id": 151645})
        (self.model / "model.safetensors").write_bytes(b"frozen weight fixture")
        self.rows = [{"benchmark": "aime24", "source_id": str(index), "answers": ["1"], "source_answers": ["001"], "prompt_token_ids": [1, index + 2], "prompt_sha256": digest([1, index + 2]), "heldout": index == 0} for index in range(2)]
        self.versions = {name: "test-1" for name in runner.REQUIRED_PACKAGES}
        self.cell = {
            "schema_version": 1, "availability": "pinned", "model": {"revision": "a" * 40},
            "model_id": "qwen25-math-1.5b-base", "family": "qwen25-math-1.5b", "staleness": None,
            "checkpoint_step": 0, "profile": "sampled-native", "benchmark": "aime24", "budget": 8,
            "context": 64, "family_config": {"native_context": 64}, "samples": 2, "seed": 42,
            "stop_token_ids": [151643, 151645], "expected_problems": 2, "expected_responses": 4,
            "comparison_sha256": "b" * 64,
            "hardware": {"gpu_name_contains": "A100", "minimum_memory_gib": 75},
            "profile_config": {"context_kind": "native", "mode": "nonthinking", "samples": 2, "temperature": 0.6, "top_p": 1.0, "top_k": -1},
            "engine": {"dtype": "bfloat16", "tensor_parallel_size": 1, "gpu_memory_utilization": 0.85, "max_num_seqs": 2, "max_num_batched_tokens": 8192, "enforce_eager": True, "enable_chunked_prefill": True, "enable_prefix_caching": False, "generation_config": "vllm"},
            "runtime": {"model_path": str(self.model), "tokenizer_path": str(self.model), "versions": self.versions},
        }
        self.save_preparation()
        self.environment = patch.dict(os.environ, {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0"}, clear=True)
        self.environment.start()
        self.metadata = patch("math_sweep.runner.importlib.metadata.version", return_value="test-1")
        self.metadata.start()
        self.grade = patch("math_sweep.scoring.grade", side_effect=lambda text, answers, unfinished_thinking=False: {"correct": not unfinished_thinking, "terminal_syntax": not unfinished_thinking, "status": "unfinished_thinking" if unfinished_thinking else "scored"})
        self.grade_mock = self.grade.start()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self.environment.stop)
        self.addCleanup(self.metadata.stop)
        self.addCleanup(self.grade.stop)

    def save_preparation(self):
        hashes = {str(path.resolve()): file_hash(path) for path in self.model.iterdir()}
        self.cell["input_files"] = hashes
        self.cell.pop("cell_id", None)
        self.cell["cell_id"] = digest(self.cell)[:24]
        write_json(self.prepared / "cell.json", self.cell)
        (self.prepared / "prompts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        write_json(self.prepared / "preparation.json", {"schema_version": 1, "files": {name: file_hash(self.prepared / name) for name in ("cell.json", "prompts.jsonl")}, "input_files": hashes, "versions": self.versions})
        self.output = self.root / "output" / self.cell["cell_id"]

    def run_engine(self, engine):
        with patch("math_sweep.runner._load_engine", return_value=(engine, Tokenizer(), SimpleNamespace, {"engine": self.cell["engine"]})):
            return runner.run_cell(self.prepared, self.root / "output")

    def test_complete_and_idempotent_with_raw_answers(self):
        engine = FakeEngine()
        receipt = self.run_engine(engine)
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["responses"], 4)
        self.assertEqual(receipt["records_sha256"], file_hash(self.output / "records.jsonl"))
        self.assertEqual(receipt["preparation_sha256"], file_hash(self.prepared / "preparation.json"))
        records = read_jsonl(self.output / "records.jsonl")
        self.assertEqual([record["heldout"] for record in records], [True, False, True, False])
        self.assertTrue(all(record["source_answers"] == ["001"] and record["raw_text"] == "Final answer: 1" for record in records))
        self.assertEqual(len(engine.batches), 2)
        self.assertTrue(all(param.stop_token_ids == [151643, 151645] and not param.ignore_eos for batch in engine.batches for param in batch))
        with patch("math_sweep.runner._load_engine", side_effect=AssertionError("must not reload completed cell")):
            self.assertEqual(runner.run_cell(self.prepared, self.root / "output"), receipt)

    def test_resume_failed_batch_without_duplicate_responses(self):
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.run_engine(FakeEngine(fail_batch=2))
        first = read_jsonl(self.output / "records.jsonl")
        self.assertEqual(len(first), 2)
        engine = FakeEngine()
        receipt = self.run_engine(engine)
        records = read_jsonl(self.output / "records.jsonl")
        self.assertEqual(records[:2], first)
        self.assertEqual(len(records), 4)
        self.assertEqual(len(engine.batches), 1)
        self.assertEqual([item["status"] for item in receipt["attempts"]], ["failed", "complete"])
        self.assertTrue((self.output / "probes-attempt-2.json").is_file())

    def test_torn_record_requires_explicit_repair(self):
        self.run_engine(FakeEngine())
        with (self.output / "records.jsonl").open("ab") as stream:
            stream.write(b'{"incomplete":')
        with self.assertRaisesRegex(ValueError, "Torn JSONL"):
            self.run_engine(FakeEngine())

    def test_changed_input_or_runtime_rejected(self):
        (self.model / "model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Frozen input hash mismatch"):
            self.run_engine(FakeEngine())
        self.save_preparation()
        with patch("math_sweep.runner.importlib.metadata.version", return_value="other"), self.assertRaisesRegex(ValueError, "Runtime version mismatch"):
            self.run_engine(FakeEngine())

    def test_native_overflow_and_extension_rejected(self):
        self.cell["context"] = 65
        self.save_preparation()
        with self.assertRaisesRegex(ValueError, "native model context"):
            self.run_engine(FakeEngine())
        self.cell["profile_config"]["context_kind"] = "unscaled_extrapolation"
        self.save_preparation()
        with self.assertRaisesRegex(NotImplementedError, "rotary-cache"):
            self.run_engine(FakeEngine())

    def test_both_eos_and_finite_gates_precede_grading(self):
        for engine, expected in [(FakeEngine(wrong_eos=True), "Forced EOS"), (FakeEngine(nonfinite=True), "Finite normal")]:
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                self.run_engine(engine)
            self.assertEqual(read_jsonl(self.output / "records.jsonl"), [])
            self.assertEqual(read_json(self.output / "receipt.json")["status"], "failed")
            self.assertEqual(self.grade_mock.call_count, 0)

    def test_receipt_and_record_provenance_cannot_be_mixed(self):
        self.run_engine(FakeEngine())
        receipt = read_json(self.output / "receipt.json")
        receipt["preparation_sha256"] = "wrong"
        write_json(self.output / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "different preparation"):
            self.run_engine(FakeEngine())

    def test_lock_prevents_concurrent_generation(self):
        self.output.mkdir(parents=True)
        with (self.output / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "already running"):
                self.run_engine(FakeEngine())

    def test_requires_allocation_and_rejects_context_override(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": ""}), self.assertRaisesRegex(ValueError, "Slurm job"):
            self.run_engine(FakeEngine())
        with patch.dict(os.environ, {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}), self.assertRaisesRegex(ValueError, "must be unset"):
            self.run_engine(FakeEngine())

    def test_runtime_rejects_wrong_gpu_family_before_engine_load(self):
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, get_device_properties=lambda index: SimpleNamespace(name="NVIDIA H100", total_memory=80 * 1024**3))
        torch = SimpleNamespace(cuda=cuda)
        vllm = SimpleNamespace(LLM=lambda **kwargs: self.fail("engine must not load on wrong hardware"), SamplingParams=SimpleNamespace)
        with patch.dict(sys.modules, {"torch": torch, "vllm": vllm}), self.assertRaisesRegex(ValueError, "frozen hardware profile"):
            runner._load_engine(self.cell, {"visible_gpu_count": 1})

    def test_thinking_mode_uses_open_prefix_decoder(self):
        self.cell["family"] = "qwen3-14b"
        self.cell["profile_config"]["mode"] = "thinking"
        self.save_preparation()
        from math_sweep.scoring import decode_completion
        with patch("math_sweep.scoring.decode_completion", wraps=decode_completion) as decoder:
            self.run_engine(FakeEngine())
        self.assertTrue(all(call.kwargs["thinking_open"] is True for call in decoder.call_args_list))
        self.assertTrue(all(record["unfinished_thinking"] for record in read_jsonl(self.output / "records.jsonl")))


if __name__ == "__main__":
    unittest.main()
