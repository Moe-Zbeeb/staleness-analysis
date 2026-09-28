import hashlib
import json
import subprocess
from pathlib import Path

root = Path('/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study')
control = root / 'launches/deepseek7b-dapo17k-b32-hp-20260929-v1'
result = json.loads((control / 'validation-complete.json').read_text())
assert set(result) == {'learner', 'worker', 'onpolicy'}
record = json.loads((control / 'submission.json').read_text())
jobs = record['jobs']
changes = []
for role, dependency in [('learner', 'afterok:2145682'), ('worker', 'after:' + jobs['learner']), ('onpolicy', 'afterok:2145682')]:
    fields = subprocess.check_output(['scontrol', 'show', 'job', '-o', jobs[role]], text=True)
    assert 'JobState=PENDING' in fields, fields
    command = ['scontrol', 'update', 'JobId=' + jobs[role], 'Dependency=' + dependency]
    subprocess.run(command, check=True)
    changes.append(command)
subprocess.run(['scancel', jobs['validation']], check=True)
receipt = {'validation': result, 'validation_execution': 'CPU-only srun step in existing high-priority allocation 2145465, using its node-local Python runtime', 'validation_source_sha256': hashlib.sha256((control / 'validate_controls.py').read_bytes()).hexdigest(), 'cancelled_redundant_cpu_job': jobs['validation'], 'dependency_updates': changes, 'fields': {}}
for role in ['learner', 'worker', 'onpolicy']:
    fields = subprocess.check_output(['scontrol', 'show', 'job', '-o', jobs[role]], text=True).strip()
    for token in ['Account=grad-students', 'Partition=high-priority', 'QOS=high-priority', 'Requeue=0']:
        assert token in fields, fields
    receipt['fields'][role] = fields
(control / 'scheduling-after-validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
