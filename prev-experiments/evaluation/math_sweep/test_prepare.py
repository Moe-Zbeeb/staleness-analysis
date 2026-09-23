import tempfile
import unittest
from pathlib import Path

from math_sweep.core import file_hash, read_json, write_json
from math_sweep.prepare import verify_model, verify_tokenizer


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.family = {"base_repo": "org/base", "base_revision": "a" * 40}
        self.model = {"staleness": 6, "checkpoint_step": 1000}
        (self.path / "model.safetensors").write_bytes(b"synthetic-weights")
        write_json(self.path / "config.json", {"max_position_embeddings": 4096})
        write_json(self.path / "training-config.json", {
            "max_off_policy_steps": 6, "training_steps": 1000, "base_model": self.family["base_repo"],
            "base_revision": self.family["base_revision"], "training_manifest_sha256": "manifest",
        })
        self.manifest = {
            "status": "passed", "step": 1000, "base_model": self.family["base_repo"],
            "base_revision": self.family["base_revision"], "training_data_sha256": "training",
            "training_manifest_sha256": "manifest",
            "validation": {"finite_tensors": True, "strict_hf_reload": True, "identical_cpu_probe_logits": True},
            "files": {name: {"sha256": file_hash(self.path / name)} for name in ("model.safetensors", "config.json", "training-config.json")},
        }
        self.save_manifest()

    def save_manifest(self):
        write_json(self.path / "export-manifest.json", self.manifest)

    def test_verified_final_export(self):
        config, hashes = verify_model(self.path, self.model, self.family, "training")
        self.assertEqual(config["max_position_embeddings"], 4096)
        self.assertIn(str((self.path / "model.safetensors").resolve()), hashes)
        with self.assertRaisesRegex(ValueError, "step-1000"):
            verify_model(self.path, {**self.model, "checkpoint_step": 999}, self.family, "training")

    def test_weight_omitted_from_manifest(self):
        del self.manifest["files"]["model.safetensors"]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "every weight shard"):
            verify_model(self.path, self.model, self.family)

    def test_phantom_weight_in_manifest(self):
        self.manifest["files"]["missing.safetensors"] = {"sha256": "missing"}
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "weight shards differ"):
            verify_model(self.path, self.model, self.family)

    def test_staleness_mismatch_even_with_valid_hash(self):
        config = read_json(self.path / "training-config.json")
        config["max_off_policy_steps"] = 4
        write_json(self.path / "training-config.json", config)
        self.manifest["files"]["training-config.json"]["sha256"] = file_hash(self.path / "training-config.json")
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "staleness"):
            verify_model(self.path, self.model, self.family)

    def test_weight_and_training_data_mismatches(self):
        with self.assertRaisesRegex(ValueError, "training data differ"):
            verify_model(self.path, self.model, self.family, "another_training_set")
        (self.path / "model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Export file mismatch"):
            verify_model(self.path, self.model, self.family, "training")

    def test_tokenizer_hash_mismatch(self):
        (self.path / "tokenizer.json").write_text("{}")
        expected = {"tokenizer.json": file_hash(self.path / "tokenizer.json")}
        verify_tokenizer(self.path, expected)
        (self.path / "tokenizer.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "Frozen tokenizer mismatch"):
            verify_tokenizer(self.path, expected)


if __name__ == "__main__":
    unittest.main()
