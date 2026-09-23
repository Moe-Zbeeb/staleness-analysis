import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import full_node
import native_devices


class FullNodeTests(unittest.TestCase):
    def test_nine_native_selectors_bind_to_uuids_not_device_minors(self):
        visible = [{'local_ordinal': i, 'uuid': str(uuid.UUID(int=i + 1)), 'pci_bus_id': f'0000:{i:02x}:00.0'} for i in range(9)]
        physical = {8-i: {'minor': 8-i, **entry} for i, entry in enumerate(visible)}
        result = native_devices.bind_visible_devices('0,1,2,3,4,5,6,7,8', '0-8', visible, physical)
        self.assertEqual([item['minor'] for item in result], list(reversed(range(9))))
        self.assertEqual([item['uuid'] for item in result], [item['uuid'] for item in visible])
        with self.assertRaises(ValueError):
            native_devices.bind_visible_devices('0,1,2', '0-2', visible[:3], physical)

    def exercise(self, failing=False):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            probes = []
            runs = []
            devices = [f'GPU-{uuid.UUID(int=i+1)}' for i in range(9)]
            info = {'job_id': '123', 'gpu_count': 9, 'worker_gpu_uuids': devices}

            def write(path, data):
                path.write_text(json.dumps(data))

            def probe(device):
                probes.append(device)
                if failing and device == devices[-1]:
                    raise RuntimeError('injected BF16 failure')
                return {'status': 'passed', 'gpu_uuid': device, 'bf16_backward': True}

            def run(args, sha):
                self.assertEqual(set(probes), set(devices))
                self.assertEqual(sha, full_node.FROZEN_SHA256)
                self.assertEqual(frozen.allocation()['gpu_count'], 9)
                self.assertEqual(frozen.allocation()['full_node_wrapper']['bf16_gpus_passed'], 9)
                runs.append(sha)
                return {'status': 'complete', 'completed': list(range(140))}

            frozen = SimpleNamespace(prepared_entries=lambda args, sha: (list(range(140)), {'package_sha256': sha}), write_json=write, file_hash=lambda path: 'validated-sha', run=run)
            argv = ['full_node.py']
            for name in ('frozen-source', 'plan', 'data', 'prepared-root', 'results-root', 'tokenizer-root'):
                argv += ['--'+name, str(root/name)]
            with patch('sys.argv', argv), patch.object(full_node, 'verify_package', return_value='wrapper-sha'), patch.object(full_node, 'load_dispatcher', return_value=frozen), patch.object(full_node, 'inspect_allocation', return_value=info), patch.object(full_node, 'probe_gpu', side_effect=probe):
                if failing:
                    with self.assertRaisesRegex(RuntimeError, 'injected BF16'):
                        full_node.main()
                    self.assertFalse(runs)
                else:
                    full_node.main()
                    self.assertEqual(len(runs), 1)
            receipt = json.loads(next((root/'results-root').rglob('validation.json')).read_text())
            self.assertEqual(receipt['status'], 'failed' if failing else 'passed')
            self.assertEqual(receipt['frozen_manifest_sha256'], full_node.FROZEN_SHA256)

    def test_all_nine_gpu_checks_precede_unchanged_dispatch(self):
        self.exercise()

    def test_any_gpu_failure_blocks_dispatch(self):
        self.exercise(failing=True)


if __name__ == '__main__':
    unittest.main()
