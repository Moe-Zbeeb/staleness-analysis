import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


PROJECT = Path('/mnt/nfs/home/mohamadzbib/projects/rl-infra')
PACKAGE = PROJECT / 'evaluation-runs/15b-math-normal-v2-occupancy-20260919/source'
FROZEN = PROJECT / 'evaluation-runs/15b-math-normal-v2-20260919/source'
OUTPUT = PROJECT / 'outputs/math-sweep-15b-normal-v2-20260919'
RECEIPT = PACKAGE.parent / 'submission.json'
FROZEN_SHA = '244f3e25f6dcd6108e93cb502251caf87e3cd2183e1f5088c7089362783b8f61'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=40)
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed: {result.stderr.strip()}')
    return result.stdout.strip()


def save(value):
    temporary = RECEIPT.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(RECEIPT)


def field(text, name):
    match = re.search(rf'\b{re.escape(name)}=([^ ]+)', text)
    if not match:
        raise ValueError(f'Missing Slurm field {name}')
    return match.group(1)


def gpu_count(text, name):
    entries = dict(part.split('=', 1) for part in field(text, name).split(',') if '=' in part)
    return int(entries.get('gres/gpu', '0'))


def verify_manifest(root, expected=None):
    path = root / 'source-manifest.json'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('Package manifest differs from reviewed hash')
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported package manifest')
    for name, wanted in manifest['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or hashlib.sha256((root / name).read_bytes()).hexdigest() != wanted:
            raise ValueError(f'Source hash mismatch: {name}')
    return digest


def verify_job(text, name, node, cpus, memory, gpus, dependency):
    expected = {'Account': 'grad-students', 'Partition': 'low-priority', 'QOS': 'normal', 'JobName': name, 'JobState': 'PENDING', 'ReqNodeList': node, 'NumCPUs': str(cpus), 'MinMemoryNode': memory}
    for key, value in expected.items():
        if field(text, key) != value:
            raise ValueError(f'Unexpected submitted {key}: {text}')
    if field(text, 'NumNodes') not in ('1', '1-1') or gpu_count(text, 'TRES') != gpus:
        raise ValueError('Submitted resources differ from the reviewed request')
    for job_id in dependency.split(':')[1:]:
        if not re.search(rf'\b{dependency.split(":")[0]}:{job_id}\(', field(text, 'Dependency')):
            raise ValueError('Submitted dependency differs from the reviewed request')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--wait-for-job', action='append', required=True, choices=('2142041', '2142020'))
    args = parser.parse_args()
    if os.getuid() != 29562:
        raise ValueError('Unexpected submitting user')
    with (PACKAGE.parent / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if RECEIPT.exists():
            raise ValueError('Submission receipt already exists; inspect it before any repeat')
        digest = verify_manifest(PACKAGE, args.expected_manifest_sha256)
        verify_manifest(FROZEN, FROZEN_SHA)
        queue = run(['squeue', '-h', '-u', 'mohamadzbib', '-o', '%i|%j'])
        if any(name in queue for name in ('m15bguard-c11', 'm15bguard-report', 'm15bnorm2-c11', 'm15bnorm2-report')):
            raise ValueError('An old or replacement shard/report job remains active')
        accounting = run(['sacct', '-j', '2142049,2142050,2142048', '--noheader', '--parsable2', '--format=JobIDRaw,State,ExitCode'])
        states = {parts[0]: parts[1] for line in accounting.splitlines() if len(parts := line.split('|')) >= 3}
        if any(not states.get(job, '').startswith('CANCELLED') for job in ('2142049', '2142050')):
            raise ValueError('Conflicted shard and obsolete report must be cancelled before replacement')
        if states.get('2142048') not in ('RUNNING', 'COMPLETED'):
            raise ValueError('Preserved node1 shard requires inspection')
        for shard, count in ((0, 60), (1, 80)):
            index = json.loads((OUTPUT / f'prepared/preparation-index-shard-{shard}.json').read_text())
            if index['status'] != 'complete' or len(index['cells']) != count or index['launcher_sha256'] != FROZEN_SHA:
                raise ValueError('Frozen prepared shard identity changed')
        node = run(['scontrol', 'show', 'node', '-o', 'deep-chungus-11'])
        if any(bad in field(node, 'State') for bad in ('DRAIN', 'DOWN', 'NOT_RESPONDING')):
            raise ValueError('Node11 is not schedulable')
        if gpu_count(node, 'CfgTRES') != 8 or gpu_count(node, 'CfgTRES') - gpu_count(node, 'AllocTRES') < 2:
            raise ValueError('Node11 no longer has the two authorized unallocated GPUs')
        if int(field(node, 'CPUTot')) - int(field(node, 'CPUAlloc')) < 8 or int(field(node, 'RealMemory')) - int(field(node, 'AllocMem')) < 24576:
            raise ValueError('Node11 CPU or scheduler memory capacity no longer fits')
        dependencies = {}
        for job_id in sorted(set(args.wait_for_job)):
            info = run(['scontrol', 'show', 'job', '-o', job_id])
            if not field(info, 'UserId').startswith('frankzydou(') or field(info, 'NodeList') != 'deep-chungus-11':
                raise ValueError('The selected wait dependency is not the reviewed foreign node11 job')
            dependencies[job_id] = info
        receipt = {'schema_version': 1, 'status': 'prepared', 'created_at': now(), 'priority': 'normal', 'launcher_sha256': digest, 'frozen_launcher_sha256': FROZEN_SHA, 'preserved_job': '2142048', 'replaces_jobs': ['2142049', '2142050'], 'preflight': {'node': node, 'queue': queue, 'accounting': accounting, 'wait_dependencies': dependencies}, 'jobs': {}}
        save(receipt)
        common = ['sbatch', '--parsable', '--hold', '--account=grad-students', '--partition=low-priority', '--qos=normal', '--nodes=1', '--ntasks=1', f'--chdir={PACKAGE}']
        dependency = 'afterany:' + ':'.join(dependencies)
        command = common + ['--job-name=m15bguard-c11', '--nodelist=deep-chungus-11', '--gres=gpu:a100:2', '--cpus-per-task=8', '--mem=24G', '--time=24:00:00', '--requeue', f'--dependency={dependency}', f'--output={OUTPUT}/logs/shard-1-guard-%j.log', f'--error={OUTPUT}/logs/shard-1-guard-%j.err', str(PACKAGE / 'evaluate_job.sh')]
        receipt['jobs']['shard_1'] = {'command': command, 'status': 'submitting'}
        save(receipt)
        job_id = run(command).split(';')[0]
        if not job_id.isdigit():
            raise ValueError('Submission returned no valid job ID')
        receipt['jobs']['shard_1'].update(job_id=job_id, status='held')
        save(receipt)
        live = run(['scontrol', 'show', 'job', '-d', '-o', job_id])
        verify_job(live, 'm15bguard-c11', 'deep-chungus-11', 8, '24G', 2, dependency)
        receipt['jobs']['shard_1']['verified'] = live
        save(receipt)
        report_dependency = f'afterok:2142048:{job_id}'
        command = common + ['--job-name=m15bguard-report', '--nodelist=deep-chungus-6', '--gres=none', '--cpus-per-task=2', '--mem=8G', '--time=01:00:00', f'--dependency={report_dependency}', f'--output={OUTPUT}/logs/report-guard-%j.log', f'--error={OUTPUT}/logs/report-guard-%j.err', str(FROZEN / 'report_job.sh')]
        receipt['jobs']['report'] = {'command': command, 'status': 'submitting'}
        save(receipt)
        report_id = run(command).split(';')[0]
        if not report_id.isdigit():
            raise ValueError('Report submission returned no valid job ID')
        receipt['jobs']['report'].update(job_id=report_id, status='held')
        save(receipt)
        live = run(['scontrol', 'show', 'job', '-d', '-o', report_id])
        verify_job(live, 'm15bguard-report', 'deep-chungus-6', 2, '8G', 0, report_dependency)
        receipt['jobs']['report']['verified'] = live
        save(receipt)
        for name in ('shard_1', 'report'):
            job = receipt['jobs'][name]
            run(['scontrol', 'release', job['job_id']])
            job.update(status='released', released_at=now(), live=run(['scontrol', 'show', 'job', '-d', '-o', job['job_id']]))
            save(receipt)
        receipt.update(status='submitted', finished_at=now())
        save(receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
