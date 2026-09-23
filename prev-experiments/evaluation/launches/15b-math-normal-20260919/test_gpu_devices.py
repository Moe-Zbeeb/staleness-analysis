import hashlib
import importlib.util
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("partial_eval_gpu_devices", Path(__file__).with_name("gpu_devices.py"))
gpu_devices = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gpu_devices)
GRES = "\n".join([
    "NodeName=deep-chungus-1, AutoDetect=off Name=gpu Type=a100 File=/dev/nvidia[0-7]",
    "NodeName=deep-chungus-[2-5], AutoDetect=off Name=gpu Type=a100 File=/dev/nvidia[0-8]",
    "NodeName=deep-chungus-[7,9-11], AutoDetect=off Name=gpu Type=a100 File=/dev/nvidia[0-7]",
]) + "\n"


def identifier(number):
    return f"GPU-00000000-0000-0000-0000-{number + 1:012d}"


def inventory():
    return {
        number: {
            "minor": number,
            "uuid": identifier(number),
            "pci_bus_id": f"0000:{7 - number:02x}:00.0",
            "information_path": f"/proc/driver/nvidia/gpus/{7 - number}/information",
            "information_sha256": str(number) * 64,
        }
        for number in range(8)
    }


class GpuMappingTests(unittest.TestCase):
    def test_reviewed_node_expressions_with_trailing_commas(self):
        for node in ("deep-chungus-1", "deep-chungus-11"):
            files, records = gpu_devices.gres_files(GRES, node)
            self.assertEqual(files, [f"/dev/nvidia{index}" for index in range(8)])
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["fields"]["AutoDetect"], "off")

    def test_slurm_ordinals_resolve_explicit_noncontiguous_device_minors(self):
        files = ["/dev/nvidia0", "/dev/nvidia2", "/dev/nvidia4", "/dev/nvidia7"]
        selected = gpu_devices.expected_devices([1, 3], files, inventory())
        self.assertEqual([item["minor"] for item in selected], [2, 7])
        self.assertEqual([item["uuid"] for item in selected], [identifier(2), identifier(7)])

    def test_duplicate_or_missing_allocated_uuid_rejected(self):
        files = [f"/dev/nvidia{number}" for number in range(8)]
        duplicate = inventory()
        duplicate[5]["uuid"] = duplicate[1]["uuid"]
        with self.assertRaisesRegex(ValueError, "not unique"):
            gpu_devices.expected_devices([1, 5], files, duplicate)
        missing = inventory()
        missing.pop(5)
        with self.assertRaisesRegex(ValueError, "no NVIDIA identity"):
            gpu_devices.expected_devices([1, 5], files, missing)

    def test_wrong_allocation_count_and_out_of_bounds_rejected(self):
        files = [f"/dev/nvidia{number}" for number in range(8)]
        for selected in ([1], [1, 2, 5], [1, 1], [1, 8]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                gpu_devices.expected_devices(selected, files, inventory())

    def test_unsupported_gres_mapping_rejected(self):
        for contents in (
            GRES.replace("AutoDetect=off", "AutoDetect=nvml"),
            GRES.replace("Type=a100", "Type=h100"),
            GRES.replace("File=/dev/nvidia[0-7]", "MultipleFiles=/dev/nvidia[0-7]"),
            GRES.replace("File=/dev/nvidia[0-7]", "File=/dev/nvidia[0-7] Count=7"),
            "Include=/etc/another-gres.conf\n" + GRES,
            GRES + "NodeName=deep-chungus-1, AutoDetect=off Name=gpu Type=a100 File=/dev/nvidia0\n",
        ):
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                gpu_devices.gres_files(contents, "deep-chungus-1")

    def test_proc_inventory_maps_minor_independently_of_pci_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for bus, minor in (("0000:01:00.0", 5), ("0000:02:00.0", 1)):
                directory = root / bus
                directory.mkdir()
                (directory / "information").write_text(f"GPU UUID: {identifier(minor)}\nDevice Minor: {minor}\n")
            result = gpu_devices.proc_inventory(root)
            self.assertEqual(result[5]["pci_bus_id"], "0000:01:00.0")
            self.assertEqual(result[1]["uuid"], identifier(1))
            self.assertEqual(len(result[5]["information_sha256"]), 64)
            (root / "0000:02:00.0/information").write_text(f"GPU UUID: {identifier(5)}\nDevice Minor: 1\n")
            with self.assertRaisesRegex(ValueError, "Duplicate NVIDIA GPU UUIDs"):
                gpu_devices.proc_inventory(root)

    def test_proc_inventory_missing_uuid_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "0000:01:00.0"
            directory.mkdir()
            (directory / "information").write_text("Device Minor: 1\n")
            with self.assertRaisesRegex(ValueError, "Missing GPU identity"):
                gpu_devices.proc_inventory(root)

    def test_character_device_minor_verified(self):
        device = {"device_file": "/dev/nvidia5", "minor": 5}
        with patch.object(Path, "stat", return_value=SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=os.makedev(195, 5))):
            gpu_devices.verify_device_files([device])
        with patch.object(Path, "stat", return_value=SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=os.makedev(195, 1))):
            with self.assertRaisesRegex(ValueError, "minor differs"):
                gpu_devices.verify_device_files([device])
        with patch.object(Path, "stat", return_value=SimpleNamespace(st_mode=stat.S_IFREG, st_rdev=os.makedev(195, 5))):
            with self.assertRaisesRegex(ValueError, "not a character device"):
                gpu_devices.verify_device_files([device])

    def test_duplicate_and_malformed_slurm_ids_rejected(self):
        self.assertEqual(gpu_devices.parse_ids("1,5"), [1, 5])
        self.assertEqual(gpu_devices.parse_ids("1-2"), [1, 2])
        for value in ("1,1", "GPU-deadbeef", "", "1-256", "5-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gpu_devices.parse_ids(value)


class HardwareTests(unittest.TestCase):
    def query(self, rows, processes="", node="deep-chungus-1"):
        devices = [inventory()[number] for number in (1, 5)]
        with patch.object(gpu_devices.subprocess, "run", side_effect=[SimpleNamespace(stdout=rows), SimpleNamespace(stdout=processes)]) as run:
            result = gpu_devices.query_hardware(devices, node)
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                self.assertEqual(call.args[0][1], "--id=" + ",".join(identifier(number) for number in (1, 5)))
                self.assertEqual(call.kwargs["timeout"], 30)
            return result

    def test_uuid_targeted_query_records_only_allocated_gpu_hardware_and_processes(self):
        rows = f"{identifier(5)}, NVIDIA A100-PCIE-40GB, 40536, 39000, 1536\n{identifier(1)}, NVIDIA A100-PCIE-40GB, 40536, 40530, 6\n"
        processes = f"{identifier(5)}, 1234, example, 1500\n"
        hardware, cohort = self.query(rows, processes)
        self.assertEqual(cohort, "a100-40gb")
        self.assertEqual([item["uuid"] for item in hardware], [identifier(1), identifier(5)])
        self.assertEqual(hardware[1]["compute_processes"][0]["pid"], 1234)
        self.assertEqual(hardware[0]["compute_processes"], [])
        self.assertEqual(hardware[0]["total_memory_mib"], 40536)

    def test_80gb_cohort_accepted_only_on_node11(self):
        rows = "".join(f"{identifier(number)}, NVIDIA A100-SXM4-80GB, 81151, 80000, 1151\n" for number in (1, 5))
        self.assertEqual(self.query(rows, node="deep-chungus-11")[1], "a100-80gb")
        with self.assertRaisesRegex(ValueError, "hardware cohort"):
            self.query(rows)

    def test_low_free_vram_and_incorrect_identity_rejected(self):
        rows = "".join(f"{identifier(number)}, NVIDIA A100-PCIE-40GB, 40536, 40000, 536\n" for number in (1, 5))
        for changed, message in ((rows.replace("40000", "30000"), "insufficient free VRAM"), (rows.replace(identifier(5), identifier(6)), "unallocated or duplicate"), (rows.replace(identifier(5), identifier(1)), "unallocated or duplicate"), (rows.splitlines()[0] + "\n", "omitted an allocated")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.query(changed)
        with self.assertRaisesRegex(ValueError, "process query returned an unallocated"):
            self.query(rows, f"{identifier(6)}, 1234, another_job, 1000\n")


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = Path(self.temporary.name) / "gres.conf"
        self.config.write_text(GRES)
        self.args = SimpleNamespace(gres_config=self.config, expected_gres_sha256=hashlib.sha256(GRES.encode()).hexdigest())
        self.env = {"SLURM_JOB_ID": "2142000", "SLURM_JOB_NUM_NODES": "1", "SLURM_GPUS_ON_NODE": "2", "SLURM_JOB_GPUS": "1,5", "CUDA_VISIBLE_DEVICES": "0,1"}

    def allocate(self, env=None, host="deep-chungus-1.csail.mit.edu"):
        with patch.dict(os.environ, self.env if env is None else env, clear=True), patch.object(gpu_devices.socket, "gethostname", return_value=host), patch.object(gpu_devices, "proc_inventory", return_value=inventory()), patch.object(gpu_devices, "verify_device_files") as verify, patch.object(gpu_devices, "query_hardware", return_value=([{"uuid": identifier(number)} for number in (1, 5)], "a100-40gb")):
            result = gpu_devices.allocation(self.args)
            verify.assert_called_once()
            return result

    def test_misleading_numeric_cuda_visibility_does_not_change_selection(self):
        result = self.allocate()
        self.assertEqual(result["worker_gpu_uuids"], [identifier(1), identifier(5)])
        self.assertEqual(result["original_cuda_visible_devices"], "0,1")
        self.assertEqual(result["gpu_count"], 2)
        self.assertEqual(result["slurm_gpu_ids"], [1, 5])
        self.assertEqual(result["gres_config_sha256"], self.args.expected_gres_sha256)
        self.assertEqual(self.allocate(host="deep-chungus-11")["worker_gpu_uuids"], result["worker_gpu_uuids"])

    def test_changed_gres_hash_rejected(self):
        self.config.write_text(GRES + "\n")
        with self.assertRaisesRegex(ValueError, "differs from reviewed"):
            self.allocate()

    def test_wrong_job_count_and_node_rejected(self):
        for key, value in (("SLURM_JOB_ID", ""), ("SLURM_JOB_NUM_NODES", "2"), ("SLURM_GPUS_ON_NODE", "3"), ("SLURM_JOB_GPUS", "1")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.allocate({**self.env, key: value})
        with self.assertRaisesRegex(ValueError, "outside this authorized"):
            self.allocate(host="deep-chungus-10")


if __name__ == "__main__":
    unittest.main()
