import copy
import os
import unittest
from unittest.mock import patch

from native_devices import bind_visible_devices, inspect_assignment

class NativeMappingTests(unittest.TestCase):
    def setUp(self):
        self.visible=[
            {'local_ordinal':0,'uuid':'756ddcc4-fe19-51ef-baa2-6f633b89829c','pci_bus_id':'0000:61:00.0'},
            {'local_ordinal':1,'uuid':'8ceb7e35-48da-9616-c892-129f3109918c','pci_bus_id':'0000:81:00.0'},
        ]
        self.physical={
            0:{'minor':0,**self.visible[0]},
            1:{'minor':1,**self.visible[1]},
            3:{'minor':3,'uuid':'a2515c5d-031e-cb1f-3e57-d6810637038c','pci_bus_id':'0000:01:00.0'},
            4:{'minor':4,'uuid':'8700e8c0-2ab7-4c31-3e0a-bcfc35597de8','pci_bus_id':'0000:e1:00.0'},
        }

    def test_cuda_selectors_do_not_select_same_numbered_linux_minors(self):
        actual=bind_visible_devices('3,4','3,4',self.visible,self.physical)
        self.assertEqual([d['uuid'] for d in actual],[d['uuid'] for d in self.visible])
        self.assertEqual([d['minor'] for d in actual],[0,1])
        self.assertNotIn(self.physical[3]['uuid'],{d['uuid'] for d in actual})

    def test_noncontiguous_selectors_keep_visible_identities(self):
        actual=bind_visible_devices('2,7','2,7',self.visible,self.physical)
        self.assertEqual([d['slurm_visible_selector'] for d in actual],[2,7])
        self.assertEqual([d['uuid'] for d in actual],[d['uuid'] for d in self.visible])

    def test_duplicate_and_partial_devices_fail(self):
        visible=copy.deepcopy(self.visible);visible[1]['uuid']=visible[0]['uuid']
        with self.assertRaisesRegex(ValueError,'Duplicate visible'):bind_visible_devices('3,4','3,4',visible,self.physical)
        with self.assertRaises(ValueError):bind_visible_devices('3','3',self.visible[:1],self.physical)

    def test_selector_and_pci_mismatch_fail(self):
        with self.assertRaisesRegex(ValueError,'Unexpected Slurm selector'):bind_visible_devices('3,4','3,5',self.visible,self.physical)
        visible=copy.deepcopy(self.visible);visible[0]['pci_bus_id']='0000:ff:00.0'
        with self.assertRaisesRegex(ValueError,'PCI identity'):bind_visible_devices('3,4','3,4',visible,self.physical)

    def test_changed_assignment_fails_before_driver_query(self):
        env={'CUDA_VISIBLE_DEVICES':'0,1','RECOVERY_SLURM_CUDA_VISIBLE_DEVICES':'3,4','SLURM_JOB_GPUS':'3,4'}
        with patch.dict(os.environ,env,clear=True),patch('native_devices.socket.gethostname',return_value='deep-chungus-11'),patch('native_devices.visible_inventory') as query:
            with self.assertRaisesRegex(ValueError,'Original Slurm CUDA_VISIBLE_DEVICES changed'):inspect_assignment()
            query.assert_not_called()

if __name__=='__main__':unittest.main()
