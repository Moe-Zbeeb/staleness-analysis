import json,unittest,uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from gpu_devices import GPU_COUNT,expected_devices,gres_files,numeric_selectors,verify_idle_devices

def identifier(n):
    return str(uuid.UUID(int=n+1))

class MappingTests(unittest.TestCase):
    def setUp(self):
        self.files=[f'/dev/nvidia{n}' for n in range(8)]
        self.physical={n:{'minor':n,'uuid':identifier(n)} for n in range(8)}
        self.inventory=[{'cuda_ordinal':i,'uuid':identifier(n)} for i,n in enumerate([4,1,0,6,2,7,3,5])]

    def test_six_devices_exclude_other_job_and_assign_all_roles(self):
        expected=expected_devices([1,2,3,4,5,6],self.files,self.physical)
        selected=numeric_selectors(expected,self.inventory)
        self.assertEqual(selected,[1,4,6,0,7,3])
        self.assertFalse(set(selected)&{2,5})
        prime=selected[4:]+selected[:4]
        self.assertEqual(prime,[7,3,1,4,6,0])
        self.assertEqual(len(prime[:2]),2)
        self.assertEqual(len(prime[2:]),4)
        self.assertFalse(set(prime[:2])&set(prime[2:]))

    def test_noncontiguous_filtered_enumeration(self):
        expected=expected_devices([0,2,3,4,6,7],self.files,self.physical)
        filtered=[{'cuda_ordinal':i,'uuid':identifier(n)} for i,n in enumerate([6,0,7,4,2,3])]
        self.assertEqual(numeric_selectors(expected,filtered),[1,4,5,3,0,2])

    def test_rejects_wrong_count_and_missing_device(self):
        with self.assertRaises(ValueError):expected_devices([0,1,2,3,4],self.files,self.physical)
        expected=expected_devices([0,1,2,3,4,5],self.files,self.physical)
        with self.assertRaises(ValueError):numeric_selectors(expected,self.inventory[:-1])

    def test_explicit_gres_and_world_size(self):
        contents='NodeName=deep-chungus-[7,9-11], AutoDetect=off Name=gpu Type=a100 File=/dev/nvidia[0-7]\nNodeName=deep-h-[1-3], AutoDetect=off Name=gpu Type=h100 File=/dev/nvidia[0-7]\n'
        for node in ['deep-chungus-9','deep-h-3']:
            self.assertEqual(gres_files(contents,node)[0],self.files)
        self.assertEqual(GPU_COUNT,6)
        self.assertEqual(GPU_COUNT*(GPU_COUNT+1)//2,21)

    def test_queries_only_allocated_uuids_for_both_hardware_types(self):
        devices=[{'uuid':identifier(n)} for n in range(6)]
        for name in ['NVIDIA A100 80GB PCIe','NVIDIA H100 80GB HBM3']:
            memory=''.join(f'GPU-{d["uuid"]}, {name}, 81920, 80000\n' for d in devices)
            with patch('gpu_devices.subprocess.run',side_effect=[SimpleNamespace(stdout=memory),SimpleNamespace(stdout='')]) as command:
                self.assertEqual(verify_idle_devices(devices)['status'],'passed')
            for call in command.call_args_list:
                self.assertEqual(call.args[0][1],'--id='+','.join('GPU-'+d['uuid'] for d in devices))

    def test_rejects_occupied_device(self):
        devices=[{'uuid':identifier(n)} for n in range(6)]
        memory=''.join(f'GPU-{d["uuid"]}, NVIDIA A100 80GB PCIe, 81920, 80000\n' for d in devices)
        with patch('gpu_devices.subprocess.run',side_effect=[SimpleNamespace(stdout=memory),SimpleNamespace(stdout=f'GPU-{devices[0]["uuid"]}, 123, 100\n')]),self.assertRaisesRegex(ValueError,'already have compute processes'):
            verify_idle_devices(devices)

    def test_rejects_low_capacity(self):
        for total,free in [(40960,40000),(81920,20000)]:
            with patch('gpu_devices.subprocess.run',return_value=SimpleNamespace(stdout=f'GPU-{identifier(0)}, NVIDIA A100, {total}, {free}\n')),self.assertRaises(ValueError):
                verify_idle_devices([{'uuid':identifier(0)}])

    def test_launcher_delta_is_exact_and_original_health_gate_retained(self):
        root=Path(__file__).resolve().parent
        original=root.parent/'7b-six-gpu-20260919/train-stale6.sh'
        provenance=json.loads((root/'provenance.json').read_text())
        text=original.read_text()
        for change in provenance['changes']:
            self.assertEqual(text.count(change['before']),1)
            text=text.replace(change['before'],change['after'])
        self.assertEqual(text,(root/'train-stale6.sh').read_text())
        self.assertIn('--nproc-per-node=6 "$RECOVERY_ROOT/health_probe.py"',text)
        self.assertNotIn('nvidia-smi -i "$CUDA_VISIBLE_DEVICES"',text)
        self.assertIn('--resume.step "$RESUME_STEP"',text)

if __name__=='__main__':
    unittest.main()
