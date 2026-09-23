import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

PROJECT = Path('/mnt/nfs/home/mohamadzbib/projects/rl-infra')
SOURCE = PROJECT / 'evaluation-runs/3b-math-node5-20260920/source'
PACKAGE = SOURCE / 'evaluation/launches/3b-math-node5-20260920'
OUTPUT = PROJECT / 'outputs/math-sweep-3b-node5-20260920'
RECEIPT = SOURCE.parent / 'submission.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def run(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError({'command': args, 'stdout': result.stdout, 'stderr': result.stderr})
    return result.stdout.strip()


def fields(value):
    return dict(re.findall(r'(?:^|\s)([^\s=]+)=([^\s]+)', value))


def gpu_count(value):
    return int(dict(item.split('=', 1) for item in value.split(',') if '=' in item).get('gres/gpu', 0))


def save(value):
    path = RECEIPT.with_suffix('.tmp.json')
    path.write_text(json.dumps(value, indent=2) + '\n')
    path.replace(RECEIPT)


def check_job(info, role, dependency):
    data = fields(info)
    expected = {'Account': 'grad-students', 'Partition': 'low-priority', 'QOS': 'normal', 'ReqNodeList': 'deep-chungus-5' if role == 'evaluate' else 'deep-chungus-6', 'Command': str(PACKAGE / f'{role}_job.sh')}
    for key, value in expected.items():
        require(data.get(key) == value, f'{role}: unexpected {key}={data.get(key)}')
    require(data['UserId'].startswith('mohamadzbib('), 'Job ownership differs')
    require(data['NumNodes'] in ('1', '1-1'), 'Unexpected node count')
    if role == 'evaluate':
        require(data.get('TresPerNode') == 'gres:gpu:a100:3' and data['MinMemoryNode'] == '72G' and data['CPUs/Task'] == '12', 'GPU worker resources differ')
    else:
        require('gres/gpu' not in data['TRES'], 'CPU stage unexpectedly requests GPUs')
    if dependency:
        require(dependency in data['Dependency'], 'Dependency differs')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    require(os.getuid() == 29562, 'Unexpected submitter')
    with (SOURCE.parent / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not RECEIPT.exists(), 'A submission receipt exists; reconcile before retry')
        manifest = PACKAGE / 'source-manifest.json'
        require(hashlib.sha256(manifest.read_bytes()).hexdigest() == args.manifest_sha256, 'Manifest changed')
        for name, expected in json.loads(manifest.read_text())['files'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe manifest path')
            require(hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected, f'File changed: {name}')
        queue = run(['squeue', '-u', 'mohamadzbib', '-h', '-o', '%i|%j|%T|%R|%b|%q'])
        require('m3b5-' not in queue, 'A 3B node5 evaluation stage is already active')
        nodes = {}
        for node, cpus, memory, gpus in [('deep-chungus-5', 12, 73728, 3), ('deep-chungus-6', 4, 16384, 0)]:
            value = run(['scontrol', 'show', 'node', '-o', node])
            nodes[node] = value
            data = fields(value)
            require(not any(state in data['State'] for state in ('DOWN', 'DRAIN', 'NOT_RESPONDING')), f'{node} unavailable')
            require(int(data['CPUTot']) - int(data['CPUAlloc']) >= cpus and int(data['RealMemory']) - int(data['AllocMem']) >= memory, f'{node} lacks CPU or scheduler memory')
            require(gpu_count(data['CfgTRES']) - gpu_count(data.get('AllocTRES', '')) >= gpus, f'{node} lacks GPUs')
            if gpus:
                require(gpu_count(data['CfgTRES']) == 9, 'Node5 configured GPU topology changed')
        jobs = run(['squeue', '-w', 'deep-chungus-5', '-h', '-o', '%i|%u|%T|%b|%j|%q'])
        receipt = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'preflight_passed', 'source_manifest_sha256': args.manifest_sha256, 'authorization': 'Use the three free GPUs on deep-chungus-5 for additional evaluations; continue normal-priority available-GPU policy.', 'models': ['qwen25-3b-base', *[f'qwen25-3b-staleness-{cap}' for cap in (2, 4, 6, 8)]], 'cells': 140, 'responses': 48620, 'node_preflight': nodes, 'node_jobs': jobs, 'user_queue': queue, 'stages': {}}
        save(receipt)
        OUTPUT.joinpath('logs').mkdir(parents=True, exist_ok=True)
        previous = None
        for role, cpus, memory, duration in [('prepare', 4, '16G', '04:00:00'), ('evaluate', 12, '72G', '48:00:00'), ('report', 2, '8G', '01:00:00')]:
            dependency = 'afterok:' + previous if previous else None
            command = ['sbatch', '--parsable', '--hold', '--account=grad-students', '--partition=low-priority', '--qos=normal', '--nodes=1', '--ntasks=1', f'--cpus-per-task={cpus}', f'--mem={memory}', f'--time={duration}', '--requeue', f'--job-name=m3b5-{role}', f'--chdir={SOURCE}', f'--output={OUTPUT}/logs/{role}-%j.log', f'--error={OUTPUT}/logs/{role}-%j.err']
            command += ['--nodelist=deep-chungus-5', '--gres=gpu:a100:3'] if role == 'evaluate' else ['--nodelist=deep-chungus-6']
            if dependency:
                command.append('--dependency=' + dependency)
            command.append(str(PACKAGE / f'{role}_job.sh'))
            receipt['stages'][role] = {'command': command, 'dependency': dependency}
            save(receipt)
            job = run(command).split(';')[0]
            require(job.isdigit(), 'Invalid submitted job ID')
            receipt['stages'][role]['job_id'] = job
            save(receipt)
            info = run(['scontrol', 'show', 'job', '-o', job])
            data = check_job(info, role, dependency)
            require(data['JobState'] == 'PENDING' and data['Priority'] == '0', 'New stage is not held')
            receipt['stages'][role]['held_info'] = info
            save(receipt)
            previous = job
        for role, stage in receipt['stages'].items():
            run(['scontrol', 'release', stage['job_id']])
            info = run(['scontrol', 'show', 'job', '-o', stage['job_id']])
            data = check_job(info, role, stage['dependency'])
            require(data['JobState'] in ('PENDING', 'RUNNING') and int(data['Priority']) > 0, 'Stage was not released')
            stage['live_info'] = info
            save(receipt)
        receipt.update(status='submitted', submitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(receipt)
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
