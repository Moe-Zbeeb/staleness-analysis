import unittest
import uuid

from gpu_devices import expected_devices, gres_files, numeric_selectors, parse_ids


def identifier(number):
    return str(uuid.UUID(int=number + 1))


class MappingTests(unittest.TestCase):
    def test_reviewed_gres_node_expression(self):
        source = "NodeName=deep-gpu-10, AutoDetect=off Name=gpu Type=2080_ti File=/dev/nvidia[0-7]\nNodeName=deep-chungus-[7,9-11], AutoDetect=off Name=gpu Type=a100 File=/dev/nvidia[0-7]\n"
        for node in ("deep-chungus-7", "deep-chungus-9", "deep-chungus-10", "deep-chungus-11"):
            files, records = gres_files(source, node)
            self.assertEqual(files, [f"/dev/nvidia{number}" for number in range(8)])
            self.assertEqual(len(records), 1)
        with self.assertRaises(ValueError):
            gres_files(source, "deep-chungus-8")

    def test_permuted_cuda_ordinals_with_partial_allocation(self):
        physical = {number: {"minor": number, "uuid": identifier(number)} for number in range(8)}
        expected = expected_devices([1, 4, 7], [f"/dev/nvidia{number}" for number in range(8)], physical)
        inventory = [{"cuda_ordinal": ordinal, "uuid": identifier(minor)} for ordinal, minor in enumerate([4, 1, 0, 6, 2, 7, 3, 5])]
        self.assertEqual(numeric_selectors(expected, inventory), [1, 0, 5])
        selected = [inventory[index]["uuid"] for index in numeric_selectors(expected, inventory)]
        self.assertEqual(selected, [identifier(number) for number in [1, 4, 7]])
        self.assertNotEqual([inventory[index]["uuid"] for index in [1, 4, 7]], selected)

    def test_cgroup_filtered_driver_enumeration(self):
        expected = [{"uuid": identifier(number)} for number in [1, 4, 7]]
        inventory = [{"cuda_ordinal": ordinal, "uuid": identifier(minor)} for ordinal, minor in enumerate([7, 1, 4])]
        self.assertEqual(numeric_selectors(expected, inventory), [1, 2, 0])

    def test_noncontiguous_gres_device_files(self):
        source = "NodeName=node, AutoDetect=off Name=gpu Type=a100 File=/dev/nvidia[0,2-3,5-7]\n"
        files, _ = gres_files(source, "node")
        inventory = {number: {"minor": number, "uuid": identifier(number)} for number in range(8)}
        selected = expected_devices([0, 2, 4], files, inventory)
        self.assertEqual([item["minor"] for item in selected], [0, 3, 6])

    def test_unverified_gres_mapping_rejected(self):
        for source in [
            "NodeName=node AutoDetect=nvml Name=gpu Type=a100 File=/dev/nvidia[0-7]",
            "NodeName=node AutoDetect=off Name=gpu Type=a100 Count=8",
            "NodeName=node AutoDetect=off Name=gpu Type=a100 File=/dev/nvidia[0-7] Count=7",
            "NodeName=node AutoDetect=off Name=gpu File=/dev/nvidia[1,0,2]",
        ]:
            with self.assertRaises(ValueError):
                gres_files(source, "node")

    def test_missing_or_duplicate_cuda_identity_rejected(self):
        expected = [{"uuid": identifier(number)} for number in range(3)]
        for inventory in [
            [{"cuda_ordinal": 0, "uuid": identifier(0)}],
            [{"cuda_ordinal": 0, "uuid": identifier(number)} for number in range(3)],
            [{"cuda_ordinal": number, "uuid": identifier(0)} for number in range(3)],
        ]:
            with self.assertRaises(ValueError):
                numeric_selectors(expected, inventory)

    def test_allocation_id_validation(self):
        self.assertEqual(parse_ids("1,4-5"), [1, 4, 5])
        for value in ["1,1,2", "2-1", "GPU-abcd", "", "0-256"]:
            with self.assertRaises(ValueError):
                parse_ids(value)
        with self.assertRaises(ValueError):
            expected_devices([0, 1, 2], ["/dev/nvidia0", "/dev/nvidia1"], {})


if __name__ == "__main__":
    unittest.main()
