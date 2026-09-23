import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("eval_occupancy_resume", ROOT / "resume.py")
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)
FROZEN = ROOT.parent / "15b-math-normal-v2-20260919"


def allocation():
    return {
        "host": "deep-chungus-11.csail.mit.edu",
        "hardware_cohort": "a100-80gb",
        "gpu_count": 2,
        "worker_gpu_uuids": ["GPU-a", "GPU-b"],
        "job_id": "2142999",
        "restart_count": "0",
        "inventory": [{"uuid": uuid, "compute_processes": [], "free_memory_mib": 71827} for uuid in ("GPU-a", "GPU-b")],
    }


class GuardTests(unittest.TestCase):
    def test_clean_same_cohort_passes_unchanged(self):
        info = allocation()
        self.assertIs(resume.validate_idle_allocation(info), info)

    def test_every_preexisting_process_is_rejected_even_with_free_memory(self):
        for owner in ("frankzydou", "mohamadzbib", None):
            info = allocation()
            info["inventory"][1]["compute_processes"] = [{"pid": 2371116, "owner": owner}]
            with self.subTest(owner=owner), self.assertRaisesRegex(ValueError, "already has compute processes"):
                resume.validate_idle_allocation(info)

    def test_hardware_and_incomplete_identity_rejected(self):
        for key, value in (("host", "deep-chungus-1"), ("hardware_cohort", "a100-40gb"), ("gpu_count", 1), ("worker_gpu_uuids", ["GPU-a", "GPU-a"])):
            info = allocation()
            info[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                resume.validate_idle_allocation(info)
        info = allocation()
        del info["inventory"][0]["compute_processes"]
        with self.assertRaisesRegex(ValueError, "evidence is missing"):
            resume.validate_idle_allocation(info)

    def test_blocked_and_passed_receipts_preserve_evidence(self):
        source = SimpleNamespace(write_json=lambda path, value: path.write_text(json.dumps(value)))
        for occupied in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                args = SimpleNamespace(results_root=Path(temporary))
                info = allocation()
                if occupied:
                    info["inventory"][0]["compute_processes"] = [{"pid": 123}]
                    with self.assertRaisesRegex(ValueError, "already has compute"):
                        resume.gated_allocation(args, lambda options: info, "a" * 64, source)
                else:
                    result = resume.gated_allocation(args, lambda options: info, "a" * 64, source)
                    self.assertEqual(result["occupancy_guard"]["source_manifest_sha256"], "a" * 64)
                saved = json.loads((args.results_root / ".native-recovery/2142999-0/allocation.json").read_text())
                self.assertEqual(saved["status"], "blocked" if occupied else "passed")
                self.assertEqual(saved["allocation"]["inventory"], info["inventory"])
                with self.assertRaisesRegex(ValueError, "already has an occupancy"):
                    resume.gated_allocation(args, lambda options: allocation(), "a" * 64, source)

    def test_exact_frozen_source_manifest_still_matches(self):
        self.assertEqual(resume.file_hash(FROZEN / "source-manifest.json"), resume.FROZEN_MANIFEST_SHA256)
        manifest = json.loads((FROZEN / "source-manifest.json").read_text())
        for name, digest in manifest["files"].items():
            self.assertEqual(resume.file_hash(FROZEN / name), digest, name)

    def test_old_memory_gate_accepted_occupied_device_new_gate_rejects_it(self):
        spec = importlib.util.spec_from_file_location("frozen_gpu_test", FROZEN / "gpu_devices.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        uuids = ["GPU-713f3893-df91-42ab-0810-d8b881f5bbed", "GPU-8ceb7e35-48da-9616-c892-129f3109918c"]
        devices = [{"uuid": value} for value in uuids]
        rows = "".join(f"{value}, NVIDIA A100 80GB PCIe, 81920, 71827, 9329\n" for value in uuids)
        processes = f"{uuids[1]}, 2371116, python, 9314\n"
        with patch.object(module.subprocess, "run", side_effect=[SimpleNamespace(stdout=rows), SimpleNamespace(stdout=processes)]):
            inventory, cohort = module.query_hardware(devices, "deep-chungus-11")
        info = allocation()
        info.update(inventory=inventory, hardware_cohort=cohort, worker_gpu_uuids=uuids)
        with self.assertRaisesRegex(ValueError, "2371116"):
            resume.validate_idle_allocation(info)


if __name__ == "__main__":
    unittest.main()
