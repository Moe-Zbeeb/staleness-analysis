import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

control = Path(__file__).resolve().parent
spec = json.loads((control / 'storage-spec-base.json').read_text())
workspace = Path(spec['workspace'])
python = Path(spec['runtime']) / 'prime-rl/.venv/bin/python'
release = workspace / 'release'
preflight = json.loads((control / 'cpu-preflight.json').read_text())
if os.environ.get('SLURMD_NODENAME') != 'deep-chungus-10':
    raise RuntimeError('Migration preparation must run on node10')
print(json.dumps({'phase': 'waiting_for_verified_stop', 'preflight': preflight}), flush=True)
deadline = time.monotonic() + 10800
while not (control / 'stop-receipt.json').exists():
    if time.monotonic() >= deadline:
        raise TimeoutError('No verified checkpoint stop before migration deadline')
    time.sleep(5)
stop = json.loads((control / 'stop-receipt.json').read_text())
if stop.get('status') != 'launcher_stopped_at_checkpoint' or stop.get('final_committed_step') != stop.get('target_step'):
    raise RuntimeError('Checkpoint stop did not verify a lossless committed boundary')
step = stop['target_step']
source = Path(stop['checkpoint']['path'])
source_study = json.loads((control / 'source-study.json').read_text())
source = Path('/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/outputs') / Path(source_study['output_dir']).name / 'checkpoints' / source.name
while not (source / 'backup-verified.json').exists():
    if time.monotonic() >= deadline:
        raise TimeoutError('Verified shared checkpoint backup did not arrive')
    time.sleep(5)
backup_receipt = json.loads((source / 'backup-verified.json').read_text())
checkpoint_bytes = sum(row['bytes'] for row in backup_receipt['files'].values())
required_free = max(180 * 1024**3, 5 * checkpoint_bytes + 74 * 1024**3)
actual_free = shutil.disk_usage(workspace).free
if actual_free < required_free:
    raise RuntimeError(f'Insufficient local capacity before resume: {actual_free} < {required_free}')
(control / 'resume-capacity.json').write_text(json.dumps({'checkpoint_bytes': checkpoint_bytes, 'required_free': required_free, 'actual_free': actual_free, 'step': step}, indent=2) + '\n')
adapted = control / 'adapted' / f'step_{step}'
receipt = control / 'migration-receipt.json'
arguments = {
    'source-checkpoint': source,
    'destination-checkpoint': adapted,
    'source-study': control / 'source-study.json',
    'target-study': control / 'target-study.json',
    'source-identity': control / 'source-identity.json',
    'target-release': release,
    'target-baseline': control / 'study.json',
    'source-assets-receipt': control / 'source-assets.json',
    'source-data-manifest': control / 'source-train-manifest.json',
    'receipt-path': receipt,
    'target-checkpoint': workspace / 'resume' / f'step_{step}',
}
environment = dict(os.environ, PYTHONPATH=str(release / 'src'), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
command = [str(python), str(control / 'migrate_inference_checkpoint.py'), '--allow-local-retention-change']
for key, value in arguments.items():
    command.extend(['--' + key, str(value)])
print(json.dumps({'phase': 'adapting_verified_checkpoint', 'step': step, 'command': command}), flush=True)
subprocess.run(command, check=True, env=environment, timeout=7200)
spec.update(resume_checkpoint=str(adapted), resume_receipt=str(receipt), resume_expected_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
(control / 'storage-spec.json').write_text(json.dumps(spec, indent=2) + '\n')
manifest = json.loads((control / 'CONTROL_SHA256.json').read_text())
manifest['storage-spec.json'] = hashlib.sha256((control / 'storage-spec.json').read_bytes()).hexdigest()
(control / 'CONTROL_SHA256.json').write_text(json.dumps(manifest, indent=2) + '\n')
subprocess.run([str(python), str(control / 'node_local_run.py'), 'prepare-resume', '--control', str(control)], check=True, env=environment, timeout=7200)
subprocess.run([str(python), str(workspace / 'node_local_run.py'), 'resume-check', '--control', str(workspace)], check=True, env=environment, timeout=1800)
result = {'status': 'verified', 'step': step, 'node': os.environ['SLURMD_NODENAME'], 'job_id': os.environ['SLURM_JOB_ID'], 'migration_receipt_sha256': spec['resume_expected_receipt_sha256'], 'local_ready_sha256': hashlib.sha256((workspace / 'ready.json').read_bytes()).hexdigest()}
with (control / 'migration-prepared.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(result), flush=True)
