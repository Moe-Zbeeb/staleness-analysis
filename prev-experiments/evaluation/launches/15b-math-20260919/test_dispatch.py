import copy
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("dispatch.py")
SPEC = importlib.util.spec_from_file_location("math_launch_dispatch", MODULE_PATH)
dispatch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dispatch)


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def entry(self, model_id, index=0):
        source = self.root / f"prepared-{index}"
        source.mkdir()
        cell = {"cell_id": f"source-{index}", "profile": "greedy-native", "profile_config": {"id": "greedy-native", "samples": 1}, "model_id": model_id, "benchmark": "math500", "samples": 1, "expected_problems": 2, "expected_responses": 2, "budget": 3072, "comparison_sha256": "a" * 64, "input_files": {"/fake/config.json": "b" * 64}, "runtime": {"versions": {"vllm": "pinned"}}}
        dispatch.write_json(source / "cell.json", cell)
        (source / "prompts.jsonl").write_text(json.dumps({"source_id": "1", "prompt_token_ids": [1, 2], "answers": ["3"]}) + "\n" + json.dumps({"source_id": "2", "prompt_token_ids": [1, 4], "answers": ["5"]}) + "\n")
        dispatch.write_json(source / "preparation.json", {"schema_version": 1, "source_cell_id": cell["cell_id"], "files": {name: dispatch.file_hash(source / name) for name in ("cell.json", "prompts.jsonl")}, "input_files": cell["input_files"], "versions": cell["runtime"]["versions"]})
        return {"prepared_dir": str(source), "cell": cell, "source_cell_id": cell["cell_id"]}

    def test_every_physical_gpu_assigned_exactly_once(self):
        inventory = [{"index": index, "uuid": f"GPU-{index}"} for index in range(9)]
        ordered = dispatch.canonical_devices("8,0,1,2,3,4,5,6,7", inventory, 9)
        self.assertEqual(ordered, ["GPU-8", *[f"GPU-{index}" for index in range(8)]])
        self.assertEqual(dispatch.canonical_devices(",".join(ordered), inventory, 9), ordered)
        for visible, count in [("0,1,2,3,4,5,6,7", 8), ("0,1,2,3,4,5,6,7,7", 9), ("GPU-0,GPU-1", 9)]:
            with self.subTest(visible=visible), self.assertRaises(ValueError):
                dispatch.canonical_devices(visible, inventory, count)

    def test_smoke_is_private_and_does_not_mutate_production(self):
        entry = self.entry(dispatch.MODEL_IDS[0])
        original = copy.deepcopy(entry)
        source_hash = dispatch.file_hash(Path(entry["prepared_dir"]) / "preparation.json")
        smoke = dispatch.create_smoke(entry, self.root / "smoke", {"worker_gpu_uuids": ["GPU-0"]}, 0, "123-attempt")
        self.assertEqual(entry, original)
        self.assertEqual(dispatch.file_hash(Path(entry["prepared_dir"]) / "preparation.json"), source_hash)
        self.assertEqual(smoke["cell"]["expected_responses"], 1)
        self.assertEqual(len(dispatch.read_jsonl(Path(smoke["prepared_dir"]) / "prompts.jsonl")), 1)
        self.assertTrue(smoke["cell"]["profile"].startswith("launch-smoke-"))
        self.assertNotEqual(smoke["cell"]["cell_id"], entry["cell"]["cell_id"])
        self.assertNotEqual(smoke["cell"]["comparison_sha256"], entry["cell"]["comparison_sha256"])
        self.assertTrue(smoke["cell"]["smoke"]["excluded_from_production"])

    def test_full_default_plan_is_required(self):
        plan = dispatch.make_plan(*dispatch.load_catalogs(dispatch.EVALUATION), selected_models=dispatch.MODEL_IDS, selected_profiles=dispatch.PROFILES)
        self.assertEqual(len(plan["cells"]), 140)
        path = self.root / "plan.json"
        dispatch.write_json(path, plan)
        self.assertEqual(dispatch.load_plan(path), plan)
        plan["cells"].pop()
        dispatch.write_json(path, plan)
        with self.assertRaisesRegex(ValueError, "exactly the frozen"):
            dispatch.load_plan(path)

    def test_smoke_failure_prevents_every_production_cell(self):
        entries = [self.entry(model, index) for index, model in enumerate(dispatch.MODEL_IDS)]
        args = SimpleNamespace(workers_from_visible=True, results_root=self.root / "results")
        allocation = {"job_id": "123", "worker_gpu_uuids": ["GPU-0", "GPU-1"], "gpu_count": 2}
        calls = []

        def fake_run(entry, output_root, device, logs, children):
            calls.append((str(output_root), device))
            if device == "GPU-0":
                raise RuntimeError("injected smoke failure")
            return {"cell_id": entry["cell"]["cell_id"]}

        with patch.object(dispatch, "prepared_entries", return_value=(entries, {"plan_sha256": "a" * 64})), patch.object(dispatch, "allocation", return_value=allocation), patch.object(dispatch, "run_process", side_effect=fake_run):
            with self.assertRaisesRegex(RuntimeError, "injected smoke"):
                dispatch.run(args, "package-sha")
        self.assertTrue(calls)
        self.assertTrue(all("smoke-results" in output for output, device in calls))
        receipts = list(args.results_root.rglob("dispatch.json"))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(dispatch.read_json(receipts[0])["status"], "failed")

    def test_all_gpu_smoke_finishes_before_production(self):
        entries = [self.entry(model, index) for index, model in enumerate(dispatch.MODEL_IDS)]
        args = SimpleNamespace(workers_from_visible=True, results_root=self.root / "results")
        allocation = {"job_id": "123", "worker_gpu_uuids": ["GPU-0", "GPU-1"], "gpu_count": 2}
        smoke = set()
        running = set()
        lock = threading.Lock()

        def fake_run(entry, output_root, device, logs, children):
            with lock:
                self.assertNotIn(device, running)
                running.add(device)
                if "smoke-results" in str(output_root):
                    smoke.add(device)
                else:
                    self.assertEqual(smoke, {"GPU-0", "GPU-1"})
                running.remove(device)
            return {"cell_id": entry["cell"]["cell_id"]}

        with patch.object(dispatch, "prepared_entries", return_value=(entries, {"plan_sha256": "a" * 64})), patch.object(dispatch, "allocation", return_value=allocation), patch.object(dispatch, "run_process", side_effect=fake_run):
            result = dispatch.run(args, "package-sha")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["completed"]), 5)
        self.assertEqual(len({item["cell_id"] for item in result["completed"]}), 5)
        self.assertEqual(len(result["smoke"]), 2)

    def test_child_output_is_durable_and_failure_is_propagated(self):
        children = dispatch.Children()
        log = self.root / "child.log"
        children.execute([sys.executable, "-c", "print('recorded output')"], dict(os.environ), log)
        self.assertIn("recorded output", log.read_text())
        with self.assertRaisesRegex(RuntimeError, "exited 7"):
            children.execute([sys.executable, "-c", "raise SystemExit(7)"], dict(os.environ), log)
        self.assertEqual(children.processes, {})
        children.stop()
        with self.assertRaises(InterruptedError):
            children.execute([sys.executable, "-c", "pass"], dict(os.environ), log)


if __name__ == "__main__":
    unittest.main()
