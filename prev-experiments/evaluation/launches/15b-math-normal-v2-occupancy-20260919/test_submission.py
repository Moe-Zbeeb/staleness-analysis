import importlib.util
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("eval_occupancy_submit", Path(__file__).with_name("submit.py"))
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


def gpu_job():
    return 'Account=grad-students Partition=low-priority QOS=normal JobName=m15bguard-c11 JobState=PENDING ReqNodeList=deep-chungus-11 NumCPUs=8 MinMemoryNode=24G NumNodes=1-1 TRES=cpu=8,mem=24G,node=1,gres/gpu=2,gres/gpu:a100=2 Dependency=afterany:2142020(unfulfilled),afterany:2142041(unfulfilled)'


class SubmissionTests(unittest.TestCase):
    def verify_gpu(self, text):
        submit.verify_job(text, 'm15bguard-c11', 'deep-chungus-11', 8, '24G', 2, 'afterany:2142020:2142041')

    def test_exact_normal_partial_gpu_request_and_both_dependencies(self):
        self.verify_gpu(gpu_job())

    def test_changed_priority_resource_or_dependency_rejected(self):
        for old, new in (('QOS=normal', 'QOS=high-priority'), ('gres/gpu=2', 'gres/gpu=8'), ('MinMemoryNode=24G', 'MinMemoryNode=8G'), ('afterany:2142041', 'afterany:2142042'), ('ReqNodeList=deep-chungus-11', 'ReqNodeList=deep-chungus-1')):
            with self.subTest(change=(old, new)), self.assertRaises(ValueError):
                self.verify_gpu(gpu_job().replace(old, new))

    def test_report_requires_preserved_shard_and_replacement_success(self):
        text = 'Account=grad-students Partition=low-priority QOS=normal JobName=m15bguard-report JobState=PENDING ReqNodeList=deep-chungus-6 NumCPUs=2 MinMemoryNode=8G NumNodes=1-1 TRES=cpu=2,mem=8G,node=1 Dependency=afterok:2142048(unfulfilled),afterok:2142999(unfulfilled)'
        submit.verify_job(text, 'm15bguard-report', 'deep-chungus-6', 2, '8G', 0, 'afterok:2142048:2142999')
        with self.assertRaises(ValueError):
            submit.verify_job(text.replace('afterok:2142999', 'afterok:2142049'), 'm15bguard-report', 'deep-chungus-6', 2, '8G', 0, 'afterok:2142048:2142999')


if __name__ == '__main__':
    unittest.main()
