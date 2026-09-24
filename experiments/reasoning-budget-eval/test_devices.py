import unittest

from devices import select_devices


class DeviceTests(unittest.TestCase):
    def test_noncontiguous_physical_indices(self):
        mapping = {str(i): 'GPU-' + str(i) for i in range(9)}
        self.assertEqual(select_devices('0,8', mapping, 2), ['GPU-0', 'GPU-8'])

    def test_slurm_index_is_not_a_device_minor(self):
        nvidia_indices = {'0': 'GPU-bus01', '4': 'GPU-bus81', '8': 'GPU-busE1'}
        self.assertEqual(select_devices('8', nvidia_indices, 1), ['GPU-busE1'])

    def test_ranges_and_duplicate_rejection(self):
        mapping = {str(i): 'GPU-' + str(i) for i in range(9)}
        self.assertEqual(select_devices('6-8', mapping, 3), ['GPU-6', 'GPU-7', 'GPU-8'])
        with self.assertRaises(AssertionError):
            select_devices('0,0', mapping, 2)


if __name__ == '__main__':
    unittest.main()
