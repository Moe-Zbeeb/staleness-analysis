import copy
import json
import os
import re
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gpu_devices import bind_visible_devices, inspect_assignment, normalize_uuid, verify_idle_devices


def identifier(number):
    return str(uuid.UUID(int=number + 1))


class MappingTests(unittest.TestCase):
    def setUp(self):
        evidence = json.loads(Path(__file__).with_name('native-cvd-probe-2142074.json').read_text())
        self.visible = [{'local_ordinal': d['local_ordinal'], 'uuid': normalize_uuid(d['uuid']), 'pci_bus_id': d['pci_bus_id']} for d in evidence['visible_devices']]
        self.physical = {}
        for d in evidence['proc_gpu_inventory']:
            minor = int(re.search(r'Device Minor:\s*(\d+)', d['contents']).group(1))
            self.physical[minor] = {'minor': minor, 'uuid': normalize_uuid(re.search(r'GPU UUID:\s*(\S+)', d['contents']).group(1)), 'pci_bus_id': Path(d['path']).parent.name}

    def test_node9_original_selectors_exclude_the_occupied_gpu(self):
        result = bind_visible_devices('0,1,2,3,4', '0,1,2,3,4', self.visible, self.physical)
        self.assertEqual([d['slurm_visible_selector'] for d in result], [0,1,2,3,4])
        self.assertEqual([d['minor'] for d in result], [3,2,1,0,7])
        self.assertNotIn('012e7790-9bf9-122d-b030-f88e489a373e', {d['uuid'] for d in result})

    def test_noncontiguous_selectors_preserve_the_given_visible_order(self):
        result = bind_visible_devices('0,2,3,5,7', '0,2,3,5,7', self.visible, self.physical)
        self.assertEqual([d['slurm_visible_selector'] for d in result], [0,2,3,5,7])
        self.assertEqual([d['uuid'] for d in result], [d['uuid'] for d in self.visible])

    def test_inconsistent_slurm_selectors_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected Slurm selector'):
            bind_visible_devices('0,1,2,3,4', '0,1,2,3,5', self.visible, self.physical)

    def test_duplicate_uuid_rejected(self):
        visible = copy.deepcopy(self.visible)
        visible[4]['uuid'] = visible[0]['uuid']
        with self.assertRaisesRegex(ValueError, 'Duplicate visible'):
            bind_visible_devices('0,1,2,3,4', '0,1,2,3,4', visible, self.physical)

    def test_physical_pci_mismatch_rejected(self):
        visible = copy.deepcopy(self.visible)
        visible[4]['pci_bus_id'] = '0000:e1:00.0'
        with self.assertRaisesRegex(ValueError, 'PCI identity'):
            bind_visible_devices('0,1,2,3,4', '0,1,2,3,4', visible, self.physical)

    def test_altered_original_assignment_rejected_before_cuda_query(self):
        environment = {'SLURM_JOB_NUM_NODES':'1','CUDA_VISIBLE_DEVICES':'0,1,2,3,5','RECOVERY_SLURM_CUDA_VISIBLE_DEVICES':'0,1,2,3,4'}
        with patch.dict(os.environ, environment, clear=True), patch('gpu_devices.socket.gethostname', return_value='deep-chungus-9'), patch('gpu_devices.visible_inventory') as query:
            with self.assertRaisesRegex(ValueError, 'Original Slurm CUDA_VISIBLE_DEVICES changed'):
                inspect_assignment()
            query.assert_not_called()


class OccupancyTests(unittest.TestCase):
    def test_only_allocated_uuids_are_queried(self):
        devices = [{'uuid': identifier(number)} for number in range(5)]
        memory = ''.join(f'GPU-{entry["uuid"]}, NVIDIA A100 80GB PCIe, 81920, 80000\n' for entry in devices)
        outputs = [SimpleNamespace(stdout=memory), SimpleNamespace(stdout='')]
        with patch('gpu_devices.subprocess.run', side_effect=outputs) as command:
            result = verify_idle_devices(devices)
        self.assertEqual(result['status'], 'passed')
        for call in command.call_args_list:
            self.assertEqual(call.args[0][1], '--id=' + ','.join('GPU-' + entry['uuid'] for entry in devices))

    def test_existing_process_rejected_even_with_enough_free_memory(self):
        devices = [{'uuid': identifier(number)} for number in range(5)]
        memory = ''.join(f'GPU-{entry["uuid"]}, NVIDIA A100 80GB PCIe, 81920, 80000\n' for entry in devices)
        processes = f'GPU-{devices[0]["uuid"]}, 12345, 100\n'
        with patch('gpu_devices.subprocess.run', side_effect=[SimpleNamespace(stdout=memory), SimpleNamespace(stdout=processes)]), self.assertRaisesRegex(ValueError, 'already have compute processes'):
            verify_idle_devices(devices)

    def test_low_free_memory_rejected(self):
        devices = [{'uuid': identifier(0)}]
        memory = f'GPU-{devices[0]["uuid"]}, NVIDIA A100 80GB PCIe, 81920, 20000\n'
        with patch('gpu_devices.subprocess.run', return_value=SimpleNamespace(stdout=memory)), self.assertRaisesRegex(ValueError, 'insufficient free VRAM'):
            verify_idle_devices(devices)


if __name__ == "__main__":
    unittest.main()
