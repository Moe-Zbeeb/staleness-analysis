import argparse
import datetime
import fcntl
import json
import os
import re
import subprocess
from pathlib import Path

import full_node

PROJECT = Path('/mnt/nfs/home/mohamadzbib/projects/rl-infra')
PACKAGE = PROJECT / 'evaluation-runs/3b-math-node5-full-20260920/source'
FROZEN = PROJECT / 'evaluation-runs/3b-math-node5-20260920/source'
OUTPUT = PROJECT / 'outputs/math-sweep-3b-node5-20260920'
RECEIPT = PACKAGE.parent / 'submission.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def run(args):
    value = subprocess.run(args, capture_output=True, text=True, timeout=30)
    if value.returncode:
        raise RuntimeError({'command': args, 'stdout': value.stdout, 'stderr': value.stderr})
    return value.stdout.strip()


def fields(value):
    return dict(re.findall(r'(?:^|\s)([^\s=]+)=([^\s]+)', value))


def save(value):
    path = RECEIPT.with_suffix('.tmp.json')
    path.write_text(json.dumps(value, indent=2)+'\n')
    path.replace(RECEIPT)


def verify_worker(info):
    value = fields(info)
    expected = {'Account': 'grad-students', 'Partition': 'low-priority', 'QOS': 'normal', 'ReqNodeList': 'deep-chungus-5', 'TresPerNode': 'gres:gpu:a100:9', 'CPUs/Task': '36', 'MinMemoryNode': '216G', 'OverSubscribe': 'NO', 'Command': str(PACKAGE/'evaluate_job.sh')}
    for key, target in expected.items():
        require(value.get(key) == target, f'Worker {key} differs: {value.get(key)}')
    require(value['UserId'].startswith('mohamadzbib(') and value['NumNodes'] in ('1', '1-1'), 'Worker identity or node count differs')
    require('afterok:2142131' in value['Dependency'] or value['Dependency'] == '(null)', 'Worker preparation dependency differs')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    require(os.getuid() == 29562, 'Unexpected submitter')
    with (PACKAGE.parent/'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not RECEIPT.exists(), 'Submission exists; reconcile before retry')
        require(full_node.verify_package() == args.manifest_sha256, 'Full-node wrapper manifest differs')
        full_node.load_dispatcher(FROZEN)
        node = run(['scontrol','show','node','-o','deep-chungus-5'])
        data = fields(node)
        require(not any(x in data['State'] for x in ['DOWN','DRAIN','NOT_RESPONDING']), 'Node5 is unavailable')
        configured = dict(x.split('=',1) for x in data['CfgTRES'].split(','))
        require(configured.get('gres/gpu') == '9' and configured.get('gres/gpu:a100') == '9', 'Node5 GPU topology changed')
        require(int(data['CPUTot']) >= 36 and int(data['RealMemory']) >= 221184, 'Full-node CPU or memory capacity is insufficient')
        node_jobs = run(['squeue','-w','deep-chungus-5','-h','-o','%i|%u|%T|%b|%j|%q'])
        queue = run(['squeue','-u','mohamadzbib','-h','-o','%i|%j|%T|%R|%b|%q'])
        require('m3b5-full' not in queue, 'Full-node evaluation already exists')
        old = run(['scontrol','show','job','-o','2142132'])
        od = fields(old)
        require(od['JobState'] == 'PENDING' and od['UserId'].startswith('mohamadzbib(') and od['Command'] == str(FROZEN/'evaluation/launches/3b-math-node5-20260920/evaluate_job.sh'), 'Old evaluation is not the expected pending job')
        report = run(['scontrol','show','job','-o','2142133'])
        rd = fields(report)
        require(rd['JobState'] == 'PENDING' and rd['UserId'].startswith('mohamadzbib(') and rd['Dependency'] == 'afterok:2142132(unfulfilled)', 'Report identity or dependency changed')
        prep = run(['sacct','-X','-j','2142131','-n','-P','--format=JobID,State,ExitCode'])
        require('2142131|RUNNING|' in prep or '2142131|COMPLETED|0:0' in prep, 'Preparation is not running or complete')
        receipt = {'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'preflight_passed','authorization':'Expand the existing normal-priority 3B evaluation to all nine GPUs on deep-chungus-5. Preserve the same prepared cells and report.','manifest_sha256':args.manifest_sha256,'frozen_manifest_sha256':full_node.FROZEN_SHA256,'prepare_job':'2142131','old_evaluation_job':'2142132','report_job':'2142133','node':node,'node_jobs':node_jobs,'queue':queue,'old_job':old,'report_before':report,'preparation':prep}
        save(receipt)
        run(['scontrol','hold','2142132'])
        run(['scontrol','hold','2142133'])
        for job in ['2142132','2142133']:
            held = fields(run(['scontrol','show','job','-o',job]))
            require(held['JobState']=='PENDING' and held['Priority']=='0', 'An old stage could not be safely held')
        command = ['sbatch','--parsable','--hold','--account=grad-students','--partition=low-priority','--qos=normal','--nodes=1','--ntasks=1','--exclusive','--cpus-per-task=36','--mem=216G','--gres=gpu:a100:9','--nodelist=deep-chungus-5','--time=48:00:00','--requeue','--job-name=m3b5-full','--dependency=afterok:2142131',f'--chdir={FROZEN}',f'--output={OUTPUT}/logs/evaluate-full-%j.log',f'--error={OUTPUT}/logs/evaluate-full-%j.err',str(PACKAGE/'evaluate_job.sh')]
        receipt['command'] = command
        save(receipt)
        job = run(command).split(';')[0]
        require(job.isdigit(), 'Invalid submitted job ID')
        receipt.update(evaluation_job=job,status='replacement_submitted_held')
        save(receipt)
        info = run(['scontrol','show','job','-o',job])
        value = verify_worker(info)
        require(value['JobState']=='PENDING' and value['Priority']=='0','Replacement is not held')
        receipt['held_job'] = info
        save(receipt)
        run(['scontrol','update','JobId=2142133','Dependency=afterok:'+job])
        info = run(['scontrol','show','job','-o','2142133'])
        require(fields(info)['Dependency']=='afterok:'+job+'(unfulfilled)','Report dependency did not update')
        receipt['updated_report'] = info
        save(receipt)
        run(['scancel','2142132'])
        receipt['old_cancelled_accounting'] = run(['sacct','-X','-j','2142132','-n','-P','--format=JobID,State,ExitCode'])
        require('CANCELLED' in receipt['old_cancelled_accounting'],'Old evaluation cancellation is unverified')
        save(receipt)
        run(['scontrol','release',job])
        run(['scontrol','release','2142133'])
        info = run(['scontrol','show','job','-o',job])
        value = verify_worker(info)
        require(value['JobState'] in ['PENDING','RUNNING'] and int(value['Priority'])>0,'Replacement did not release')
        receipt.update(status='submitted',worker_live=info,report_live=run(['scontrol','show','job','-o','2142133']),completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(receipt)
        print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
