import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from staleness_guard import config, fingerprint, gate, read, require, static


def run(command):
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()


def fields(line):
    return dict(re.findall(r'(\w+)=([^\s]+)', line))


def candidates(output, count):
    selected = []
    for line in output.splitlines():
        node = fields(line)
        if 'high-priority' not in node.get('Partitions', '').split(','):
            continue
        if any(state in node.get('State', '') for state in ['DOWN', 'DRAIN', 'FAIL', 'UNKNOWN', 'NOT_RESPONDING']):
            continue
        gpus = re.findall(r'(?:^|,)gpu:a100:(\d+)(?:\([^)]*\))?(?:,|$)', node.get('Gres', ''))
        if len(gpus) == 1 and int(gpus[0]) == count and int(node.get('CPUTot', 0)) >= 64 and int(node.get('RealMemory', 0)) >= 393216:
            selected.append(node['NodeName'])
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--stage', choices=['smoke', 'train'], required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    static(root)
    require(str(root) == str(Path(read(root / 'comparison.json')['run_dir']).parents[2] / 'experiments' / root.name), 'Run this from the deployed cluster workspace')
    if args.stage == 'train':
        gate(root)
    count = config(root)['deployment']['gpus_per_node']
    node_output = run(['scontrol', 'show', 'nodes', '-o'])
    nodes = candidates(node_output, count)
    require(nodes, f'No healthy full A100 node has exactly {count} GPUs. Keep the frozen topology; resolve placement before submitting.')
    excluded = sorted(fields(line)['NodeName'] for line in node_output.splitlines() if fields(line).get('NodeName') not in nodes)
    qos = run(['sacctmgr', '-nP', 'show', 'qos', 'high-priority', 'format=Name,MaxTRESPU,Flags'])
    queue = run(['squeue', '-u', os.environ['USER'], '-h', '-o', '%i|%T|%R|%b|%q'])
    infra = root.parents[1]
    logs = infra / 'logs' / root.name
    command = ['sbatch', '--parsable', '--account=grad-students', '--partition=high-priority', '--qos=high-priority', '--nodes=1', '--exclusive', f'--gres=gpu:a100:{count}', '--ntasks=1', '--cpus-per-task=64', '--mem=384G', '--time=' + ('02:00:00' if args.stage == 'smoke' else '0'), '--requeue', '--signal=B:USR1@180', '--job-name=' + root.name + '-' + args.stage, '--chdir=' + str(infra), '--output=' + str(logs / (args.stage + '-%j.out')), '--error=' + str(logs / (args.stage + '-%j.err')), '--wrap=exec bash ' + shlex.quote(str(root / 'scripts' / (args.stage + '_job.sh')))]
    if excluded:
        command.insert(8, '--exclude=' + ','.join(excluded))
    record = {'captured_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_sha256': fingerprint(root), 'command': command, 'candidate_nodes': nodes, 'qos': qos, 'queue': queue, 'submitted': False}
    if args.submit:
        logs.mkdir(parents=True, exist_ok=True)
        job_id = run(command).split(';')[0]
        require(job_id.isdigit(), 'Unrecognized Slurm job ID')
        record.update(submitted=True, job_id=job_id)
        details = run(['scontrol', 'show', 'job', '-o', job_id])
        record['job'] = details
        (root / 'validation').mkdir(exist_ok=True)
        (root / 'validation' / f'submission-{job_id}.json').write_text(json.dumps(record, indent=2) + '\n')
        job = fields(details)
        for key, expected in [('Account', 'grad-students'), ('Partition', 'high-priority'), ('QOS', 'high-priority')]:
            require(job.get(key) == expected, f'Job {job_id}: unexpected {key}; inspect the recorded submission')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
