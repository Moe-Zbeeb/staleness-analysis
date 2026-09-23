import copy
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch
from gpu_devices import bind_visible_devices, verify_idle_devices

class FullNodeTests(unittest.TestCase):
    def setUp(self):
        self.visible=[{'local_ordinal':i,'uuid':str(uuid.UUID(int=i+1)),'pci_bus_id':f'0000:{i+16:02x}:00.0'} for i in range(8)]
        self.physical={7-i:{'minor':7-i,'uuid':d['uuid'],'pci_bus_id':d['pci_bus_id']} for i,d in enumerate(self.visible)}
        self.ids=','.join(str(i) for i in range(8))

    def test_original_cuda_order_survives_reversed_minor_order(self):
        result=bind_visible_devices(self.ids,self.ids,self.visible,self.physical)
        self.assertEqual([d['slurm_visible_selector'] for d in result],list(range(8)))
        self.assertEqual([d['minor'] for d in result],list(reversed(range(8))))
        self.assertEqual([d['uuid'] for d in result],[d['uuid'] for d in self.visible])

    def test_partial_allocation_rejected(self):
        with self.assertRaises(ValueError):bind_visible_devices('0,1,2,3,4','0,1,2,3,4',self.visible[:5],self.physical)

    def test_duplicate_physical_device_rejected(self):
        values=copy.deepcopy(self.visible);values[7]['uuid']=values[0]['uuid']
        with self.assertRaisesRegex(ValueError,'Duplicate visible'):bind_visible_devices(self.ids,self.ids,values,self.physical)

    def test_pci_mismatch_rejected(self):
        values=copy.deepcopy(self.visible);values[7]['pci_bus_id']='0000:ff:00.0'
        with self.assertRaisesRegex(ValueError,'PCI identity'):bind_visible_devices(self.ids,self.ids,values,self.physical)

    def test_existing_process_rejected(self):
        memory=''.join(f'GPU-{d["uuid"]}, NVIDIA A100 80GB PCIe, 81920, 81000\n' for d in self.visible)
        process=f'GPU-{self.visible[7]["uuid"]}, 123, 100\n'
        with patch('gpu_devices.subprocess.run',side_effect=[SimpleNamespace(stdout=memory),SimpleNamespace(stdout=process)]),self.assertRaisesRegex(ValueError,'already have compute processes'):
            verify_idle_devices(self.visible)

if __name__=='__main__':unittest.main()
