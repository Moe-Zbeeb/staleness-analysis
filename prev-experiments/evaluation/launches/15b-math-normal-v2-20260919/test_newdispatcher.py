import copy
import fcntl
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
SPEC = importlib.util.spec_from_file_location("partial_math_dispatch", PACKAGE / "newdispatcher.py")
dispatch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dispatch)
SOURCE_SPEC = importlib.util.spec_from_file_location("test_original_dispatch", PACKAGE.parent / "15b-math-20260919/dispatch.py")
source = importlib.util.module_from_spec(SOURCE_SPEC)
SOURCE_SPEC.loader.exec_module(source)


class PartialDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.plan = source.make_plan(*source.load_catalogs(source.EVALUATION), selected_models=dispatch.MODEL_IDS, selected_profiles=["greedy-native", "sampled-native"])
        self.entries = [{"cell": cell, "source_cell_id": cell["cell_id"], "prepared_dir": "unused"} for cell in self.plan["cells"]]

    def prepared(self, model_id):
        cell = copy.deepcopy(next(entry["cell"] for entry in self.entries if entry["cell"]["model_id"] == model_id and entry["cell"]["profile"] == "greedy-native" and entry["cell"]["benchmark"] == "math500"))
        source_id = cell.pop("cell_id")
        cell.update(comparison_sha256="a" * 64, input_files={"/immutable/code.py": "b" * 64}, runtime={"model_path": "/immutable/model", "versions": {"vllm": "pinned"}})
        cell["cell_id"] = source.digest(cell)[:24]
        directory = self.root / "original" / cell["cell_id"]
        directory.mkdir(parents=True)
        source.write_json(directory / "cell.json", cell)
        (directory / "prompts.jsonl").write_text(json.dumps({"benchmark": "math500", "source_id": "1", "prompt_token_ids": [1, 2], "answers": ["3"], "heldout": True}) + "\n")
        source.write_json(directory / "preparation.json", {"schema_version": 1, "source_cell_id": source_id, "files": {name: source.file_hash(directory / name) for name in ("cell.json", "prompts.jsonl")}, "input_files": cell["input_files"], "versions": cell["runtime"]["versions"]})
        return {"cell": cell, "source_cell_id": source_id, "prepared_dir": str(directory)}

    def test_partition_is_complete_disjoint_and_group_preserving(self):
        shards, specification = dispatch.partition_entries(self.entries, 2)
        self.assertEqual(sum(map(len, shards)), 140)
        identifiers = [{entry["cell"]["cell_id"] for entry in items} for items in shards]
        self.assertFalse(identifiers[0] & identifiers[1])
        self.assertEqual(len(identifiers[0] | identifiers[1]), 140)
        owners = {}
        for shard, values in enumerate(shards):
            for entry in values:
                key = entry["cell"]["profile"], entry["cell"]["benchmark"]
                self.assertEqual(owners.setdefault(key, shard), shard)
        self.assertEqual(len(specification["groups"]), 20)
        self.assertEqual(dispatch.partition_entries(list(reversed(self.entries)), 2)[1], specification)
        self.assertEqual(specification["gpu_counts"], {"0": 1, "1": 2})
        loads = specification["maximum_output_tokens_per_gpu"]
        self.assertLess(abs(loads[0] - loads[1]) / sum(loads), 0.10)
        with self.assertRaises(ValueError):
            dispatch.partition_entries(self.entries, 3)
        with self.assertRaises(ValueError):
            dispatch.partition_entries(self.entries[:-1], 2)

    def test_derivation_only_changes_declared_metadata_and_keeps_original_bytes(self):
        entry = self.prepared(dispatch.MODEL_IDS[0])
        original = copy.deepcopy(entry)
        files = {path.name: path.read_bytes() for path in Path(entry["prepared_dir"]).iterdir()}
        derived = dispatch.derive_entry(entry, self.root / "derived", 0, "launcher-sha", source)
        changed = {key for key in set(entry["cell"]) | set(derived["cell"]) if entry["cell"].get(key) != derived["cell"].get(key)}
        self.assertEqual(changed, {"hardware", "comparison_sha256", "transition", "cell_id"})
        self.assertEqual(entry, original)
        self.assertEqual(files, {path.name: path.read_bytes() for path in Path(entry["prepared_dir"]).iterdir()})
        directory = Path(derived["prepared_dir"])
        self.assertEqual((directory / "prompts.jsonl").read_bytes(), files["prompts.jsonl"])
        self.assertEqual(source.read_json(directory / "preparation.json")["input_files"], entry["cell"]["input_files"])
        self.assertEqual(derived["cell"]["hardware"]["minimum_memory_gib"], 37)
        self.assertEqual(dispatch.derive_entry(entry, self.root / "derived", 0, "launcher-sha", source), derived)
        (directory / "prompts.jsonl").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "prepared files changed"):
            dispatch.derive_entry(entry, self.root / "derived", 0, "launcher-sha", source)

    def test_comparison_hash_matches_model_arms_only_within_same_hardware_cohort(self):
        base = self.prepared(dispatch.MODEL_IDS[0])
        trained = self.prepared(dispatch.MODEL_IDS[1])
        left = dispatch.derive_entry(base, self.root / "derived", 0, "launcher", source)
        right = dispatch.derive_entry(trained, self.root / "derived", 0, "launcher", source)
        other_node = dispatch.derive_entry(base, self.root / "derived", 1, "launcher", source)
        self.assertEqual(left["cell"]["comparison_sha256"], right["cell"]["comparison_sha256"])
        self.assertNotEqual(left["cell"]["comparison_sha256"], other_node["cell"]["comparison_sha256"])
        self.assertEqual(other_node["cell"]["hardware"]["minimum_memory_gib"], 75)

    def test_each_node_smokes_all_five_models_before_production(self):
        entries = [self.prepared(model) for model in dispatch.MODEL_IDS]
        partition = [[entries[0], entries[1]], entries[2:]]
        specification = {"groups": [], "shards": 2}
        args = SimpleNamespace(workers_from_visible=True, shard=0, shards=2, original_results_root=self.root / "results", results_root=self.root / "results", derived_prepared_root=self.root / "derived")
        allocation = {"job_id": "123", "host": "deep-chungus-1", "gpu_count": 1, "worker_gpu_uuids": ["GPU-A"]}
        smoked = set()

        def fake_run(entry, output_root, device, logs, children):
            if "smoke-results" in str(output_root):
                smoked.add(entry["cell"]["model_id"])
            else:
                self.assertEqual(smoked, set(dispatch.MODEL_IDS))
                self.assertEqual(entry["cell"]["hardware"]["minimum_memory_gib"], 37)
            return {"cell_id": entry["cell"]["cell_id"], "responses": 1}

        with patch.object(source, "prepared_entries", return_value=(entries, {"plan_sha256": "a" * 64})), patch.object(dispatch, "partition_entries", return_value=(partition, specification)), patch("gpu_devices.allocation", return_value=allocation), patch.object(source, "run_process", side_effect=fake_run):
            result = dispatch.run(args, source, "source-sha", "launcher-sha")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["smoke"]), 5)
        self.assertEqual(len(result["completed"]), 2)
        self.assertTrue((args.derived_prepared_root / "preparation-index-shard-0.json").exists())
        self.assertEqual(len(list(args.derived_prepared_root.glob("*/cell.json"))), 2)

    def test_shared_lock_descriptor_is_readable_for_nfs(self):
        args = SimpleNamespace(workers_from_visible=True, shard=0, shards=2, original_results_root=self.root / "original-results", results_root=self.root / "results")
        allocation = {"job_id": "123", "host": "deep-chungus-1", "gpu_count": 1, "worker_gpu_uuids": ["GPU-A"]}
        actual_flock = fcntl.flock
        shared_checks = []

        def nfs_flock(stream, operation):
            if operation & fcntl.LOCK_SH:
                access = fcntl.fcntl(stream.fileno(), fcntl.F_GETFL) & os.O_ACCMODE
                if access == os.O_WRONLY:
                    raise OSError(9, "Shared NFS lock requires a readable descriptor")
                shared_checks.append(access)
            return actual_flock(stream, operation)

        with patch.object(source, "prepared_entries", return_value=(self.entries, {"plan_sha256": "a" * 64})), patch("gpu_devices.allocation", return_value=allocation), patch.object(dispatch.fcntl, "flock", side_effect=nfs_flock), patch.object(source, "write_json", side_effect=RuntimeError("reached partition after shared lock")):
            with self.assertRaisesRegex(RuntimeError, "reached partition after shared lock"):
                dispatch.run(args, source, "source-sha", "launcher-sha")
        self.assertEqual(shared_checks, [os.O_RDWR])

    def test_node1_rejects_two_gpu_allocation(self):
        args = SimpleNamespace(workers_from_visible=True, shard=0, shards=2)
        allocation = {"job_id": "123", "host": "deep-chungus-1", "gpu_count": 2, "worker_gpu_uuids": ["GPU-A", "GPU-B"]}
        with patch.object(source, "prepared_entries", return_value=(self.entries, {"plan_sha256": "a" * 64})), patch("gpu_devices.allocation", return_value=allocation):
            with self.assertRaisesRegex(ValueError, "exactly 1 distinct allocated GPUs"):
                dispatch.run(args, source, "source-sha", "launcher-sha")

    def test_original_full_matrix_dispatch_lock_blocks_partial_dispatch(self):
        args = SimpleNamespace(workers_from_visible=True, shard=0, shards=2, original_results_root=self.root / "results", results_root=self.root / "results")
        allocation = {"job_id": "123", "host": "deep-chungus-1", "gpu_count": 1, "worker_gpu_uuids": ["GPU-A"]}
        scope = args.results_root / ".dispatch" / ("a" * 24)
        scope.mkdir(parents=True)
        with (scope / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(source, "prepared_entries", return_value=(self.entries, {"plan_sha256": "a" * 64})), patch("gpu_devices.allocation", return_value=allocation), self.assertRaises(BlockingIOError):
                dispatch.run(args, source, "source-sha", "launcher-sha")


if __name__ == "__main__":
    unittest.main()
