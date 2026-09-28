import hashlib
import json
import os
import shutil
from pathlib import Path

workspace = Path('/tmp/mohamadzbib-staleness-storage-v2/deepseek15b-dapo17k-6k-exact256-b32-seed42-v1')
if os.environ.get('SLURM_JOB_ID') != '2145464':
    raise ValueError('Wrong training allocation')
source = Path('/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/hotfixes/backup-20260928/scripts')
target = workspace / 'backup-fix-20260928'
target.mkdir(exist_ok=False)
checksums = {}
for name in ('local_backup.py', 'backup_handoff.py'):
    shutil.copy2(source / name, target / name)
    expected = hashlib.sha256((source / name).read_bytes()).hexdigest()
    if hashlib.sha256((target / name).read_bytes()).hexdigest() != expected:
        raise ValueError('Hotfix staging checksum mismatch')
    checksums[name] = expected
(target / 'source-checksums.json').write_text(json.dumps(checksums, indent=2) + '\n')
os.execv('/usr/bin/python3', ['/usr/bin/python3', str(target / 'backup_handoff.py'), '--supervisor', str(workspace / 'supervisor.json'), '--script', str(target / 'local_backup.py'), '--receipt', str(target / 'handoff.json')])
