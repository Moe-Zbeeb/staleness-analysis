import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

PROJECT=Path('/mnt/nfs/home/mohamadzbib/projects/rl-infra')
PACKAGE=PROJECT/'evaluation-runs/15b-math-normal-v2-native-20260920/source'
FROZEN=PROJECT/'evaluation-runs/15b-math-normal-v2-20260919/source'
OUTPUT=PROJECT/'outputs/math-sweep-15b-normal-v2-20260919'
RECEIPT=PACKAGE.parent/'submission.json'
FROZEN_SHA='244f3e25f6dcd6108e93cb502251caf87e3cd2183e1f5088c7089362783b8f61'

def run(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=30)
    if p.returncode:raise RuntimeError({'command':args,'stderr':p.stderr,'stdout':p.stdout})
    return p.stdout.strip()

def fields(value):
    return dict(re.findall(r'(\w+)=([^\s]+)',value))

def require(ok,message):
    if not ok:raise ValueError(message)

def save(receipt):
    temporary=RECEIPT.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(receipt,indent=2)+'\n');temporary.replace(RECEIPT)

def verify_manifest(path,expected):
    manifest=path/'source-manifest.json'
    require(hashlib.sha256(manifest.read_bytes()).hexdigest()==expected,'Manifest identity changed')
    for name,digest in json.loads(manifest.read_text())['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Invalid manifest path')
        require(hashlib.sha256((path/name).read_bytes()).hexdigest()==digest,f'File changed: {name}')

def count(value):
    return int(dict(p.split('=',1) for p in value.split(',') if '=' in p).get('gres/gpu',0))

def verify_worker(info):
    d=fields(info)
    for key,value in {'Account':'grad-students','Partition':'low-priority','QOS':'normal','ReqNodeList':'deep-chungus-11','TresPerNode':'gres:gpu:a100:2','MinMemoryNode':'24G','CPUs/Task':'8','Dependency':'(null)'}.items():
        if key=='CPUs/Task':continue
        require(d[key]==value,f'Worker {key} differs: {d.get(key)}')
    require(d['UserId'].startswith('mohamadzbib('),'Unexpected job owner')
    require(d['NumNodes'] in ['1','1-1'],'Unexpected node count')
    return d

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest-sha256',required=True);args=parser.parse_args()
    require(os.getuid()==29562,'Unexpected submitter')
    with (PACKAGE.parent/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not RECEIPT.exists(),'Submission exists; reconcile before retry')
        verify_manifest(PACKAGE,args.manifest_sha256);verify_manifest(FROZEN,FROZEN_SHA)
        queue=run(['squeue','-u','mohamadzbib','-h','-o','%i|%j|%T|%R|%b'])
        require(not any(x in queue for x in ['m15bnative-c11','m15bguard-c11','m15bnorm2-c11']),'An evaluation worker is already active')
        accounting=run(['sacct','-X','-j','2142048,2142053','-n','-P','--format=JobID,State,ExitCode'])
        require('2142048|COMPLETED|0:0' in accounting and '2142053|FAILED|1:0' in accounting,'Previous worker states changed')
        indexes={i:json.loads((OUTPUT/f'prepared/preparation-index-shard-{i}.json').read_text()) for i in [0,1]}
        for i,n in [(0,60),(1,80)]:
            require(indexes[i]['status']=='complete' and len(indexes[i]['cells'])==n and indexes[i]['launcher_sha256']==FROZEN_SHA,'Prepared cells differ')
        groups={i:{v['cell_id'] for v in indexes[i]['cells'].values()} for i in [0,1]}
        require(not groups[0]&groups[1],'Shard cells overlap')
        preserved=[]
        for cell in sorted(groups[0]):
            p=OUTPUT/'results'/cell/'receipt.json';data=json.loads(p.read_text())
            require(data['status']=='complete' and data['responses']==data['expected_responses'],'A preserved result is incomplete')
            preserved.append({'cell_id':cell,'responses':data['responses'],'receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        node=run(['scontrol','show','node','-o','deep-chungus-11']);d=fields(node)
        require(not any(x in d['State'] for x in ['DOWN','DRAIN','NOT_RESPONDING']),'Node11 unavailable')
        require(count(d['CfgTRES'])==8 and count(d['CfgTRES'])-count(d.get('AllocTRES',''))>=2,'Fewer than two free node11 GPUs')
        require(int(d['CPUTot'])-int(d['CPUAlloc'])>=8 and int(d['RealMemory'])-int(d['AllocMem'])>=24576,'Insufficient scheduler CPU or memory capacity')
        report=run(['scontrol','show','job','-o','2142054']);rd=fields(report)
        require(rd['JobState']=='PENDING' and rd['UserId'].startswith('mohamadzbib(') and rd['QOS']=='normal' and rd['Command']==str(FROZEN/'report_job.sh'),'Existing report identity changed')
        require('2142053' in rd['Dependency'],'Report dependency changed')
        receipt={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'preflight_passed','authorization':'Continue the 1.5B base and staleness 2/4/6/8 evaluation on deep-chungus-11; preserve the prior normal-priority two-free-GPU authorization.','source_manifest_sha256':args.manifest_sha256,'frozen_launcher_sha256':FROZEN_SHA,'previous_worker':'2142053','preserved_worker':'2142048','preserved_cells':preserved,'remaining_shard_cells':sorted(groups[1]),'report_job_id':'2142054','node':node,'queue':queue,'accounting':accounting,'report_before':report}
        save(receipt)
        command=['sbatch','--parsable','--hold','--account=grad-students','--partition=low-priority','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=24G','--gres=gpu:a100:2','--nodelist=deep-chungus-11','--job-name=m15bnative-c11','--time=24:00:00','--requeue',f'--chdir={PACKAGE}',f'--output={OUTPUT}/logs/shard-1-native-%j.log',f'--error={OUTPUT}/logs/shard-1-native-%j.err',str(PACKAGE/'evaluate_job.sh')]
        receipt['worker_command']=command;save(receipt)
        job=run(command).split(';')[0];require(job.isdigit(),'Invalid submitted job ID')
        receipt.update(worker_job_id=job,status='worker_submitted_held');save(receipt)
        worker=run(['scontrol','show','job','-o',job]);wd=verify_worker(worker)
        require(wd['JobState']=='PENDING' and wd['Priority']=='0','New worker not held')
        receipt['worker_held']=worker;save(receipt)
        run(['scontrol','hold','2142054'])
        require(fields(run(['scontrol','show','job','-o','2142054']))['JobState']=='PENDING','Report unexpectedly started')
        dependency='afterok:'+job
        run(['scontrol','update','JobId=2142054','Dependency='+dependency])
        report=run(['scontrol','show','job','-o','2142054']);rd=fields(report)
        require(rd['Dependency']==dependency+'(unfulfilled)' and rd['Priority']=='0','Report dependency update differs')
        receipt['report_updated']=report;receipt['status']='dependencies_updated';save(receipt)
        for job_id in [job,'2142054']:run(['scontrol','release',job_id])
        worker=run(['scontrol','show','job','-o',job]);wd=verify_worker(worker)
        require(wd['JobState'] in ['PENDING','RUNNING'] and int(wd['Priority'])>0,'Worker did not release')
        report=run(['scontrol','show','job','-o','2142054']);rd=fields(report)
        require(rd['JobState']=='PENDING' and int(rd['Priority'])>0 and job in rd['Dependency'],'Report release differs')
        receipt.update(status='submitted',worker_live=worker,report_live=report,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());save(receipt)
        print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
