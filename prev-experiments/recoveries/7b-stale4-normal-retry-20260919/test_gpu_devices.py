import unittest
import uuid

from gpu_devices import GPU_COUNT, expected_devices, gres_files, numeric_selectors


def identifier(number):
    return str(uuid.UUID(int=number + 1))


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.files = [f"/dev/nvidia{number}" for number in range(8)]
        self.physical = {number: {"minor": number, "uuid": identifier(number)} for number in range(8)}
        self.inventory = [{"cuda_ordinal": ordinal, "uuid": identifier(minor)} for ordinal, minor in enumerate([4, 1, 0, 6, 2, 7, 3, 5])]

    def test_five_gpu_partition_does_not_overlap_three_gpu_job(self):
        expected = expected_devices([3, 4, 5, 6, 7], self.files, self.physical)
        selected = numeric_selectors(expected, self.inventory)
        self.assertEqual(selected, [6, 0, 7, 3, 5])
        other_job = {entry["cuda_ordinal"] for entry in self.inventory if entry["uuid"] in {identifier(number) for number in [0, 1, 2]}}
        self.assertEqual(other_job, {1, 2, 4})
        self.assertFalse(set(selected) & other_job)
        prime = selected[4:] + selected[:4]
        self.assertEqual(prime, [5, 6, 0, 7, 3])
        self.assertEqual(set(prime[:1]) & set(prime[1:]), set())
        self.assertEqual(set(prime), set(selected))

    def test_noncontiguous_allocation_and_filtered_enumeration(self):
        expected = expected_devices([0, 2, 4, 6, 7], self.files, self.physical)
        self.assertEqual(numeric_selectors(expected, self.inventory), [2, 4, 0, 3, 5])
        filtered = [{"cuda_ordinal": ordinal, "uuid": identifier(minor)} for ordinal, minor in enumerate([6, 0, 7, 4, 2])]
        self.assertEqual(numeric_selectors(expected, filtered), [1, 4, 3, 0, 2])

    def test_wrong_count_or_missing_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            expected_devices([0, 1, 2], self.files, self.physical)
        expected = expected_devices([3, 4, 5, 6, 7], self.files, self.physical)
        with self.assertRaises(ValueError):
            numeric_selectors(expected, [entry for entry in self.inventory if entry["uuid"] != identifier(7)])

    def test_actual_gres_node_expression(self):
        source = "NodeName=deep-chungus-[7,9-11], AutoDetect=off Name=gpu Type=a100 File=/dev/nvidia[0-7]\n"
        self.assertEqual(gres_files(source, "deep-chungus-10")[0], self.files)
        self.assertEqual(GPU_COUNT, 5)
        self.assertEqual(GPU_COUNT * (GPU_COUNT + 1) / 2, 15)


if __name__ == "__main__":
    unittest.main()
