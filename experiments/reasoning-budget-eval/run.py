import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import ROOT, validate_manifest, write_json
from devices import select_devices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-index', type=int, required=True)
    args = parser.parse_args()
    validate_manifest()
    spec = json.loads((ROOT / 'spec.json').read_text())
    model = spec['models'][args.model_index]
    output = ROOT / 'outputs' / model['tag']
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader,nounits'], text=True, timeout=30).splitlines()
    gpu_info = {}
    by_index = {}
    for row in rows:
        index, uuid, name, memory = [v.strip() for v in row.split(',')]
        by_index[index] = uuid
        gpu_info[uuid] = {'name': name, 'memory_mib': int(memory)}
    allocated = os.environ['SLURM_JOB_GPUS']
    devices = select_devices(allocated, by_index, spec['replicas'])
    assert len(devices) == 8 and set(devices) == set(by_index.values()), 'This launcher requires one full eight-GPU node'
    assert all('A100' in gpu_info[g]['name'] and gpu_info[g]['memory_mib'] >= 79000 for g in devices)
    write_json(output / 'allocation.json', {'job_id': os.environ['SLURM_JOB_ID'], 'slurm_job_gpus': allocated, 'original_cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'worker_uuids': devices, 'gpu_info': gpu_info, 'node': os.environ.get('SLURMD_NODENAME'), 'whole_node_verified': True})
    processes = []
    def stop(*unused):
        for proc in processes:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        raise SystemExit(1)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for slot, device in enumerate(devices):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = device
            cache = Path('/tmp') / ('reasoning-rollouts-' + os.environ['SLURM_JOB_ID']) / str(slot)
            cache.mkdir(parents=True, exist_ok=True)
            env.update(VLLM_CACHE_ROOT=str(cache / 'vllm'), TORCHINDUCTOR_CACHE_DIR=str(cache / 'torchinductor'), TRITON_CACHE_DIR=str(cache / 'triton'))
            log = (output / f'worker-{slot}-{os.environ["SLURM_JOB_ID"]}.log').open('w')
            processes.append(subprocess.Popen([sys.executable, str(ROOT / 'worker.py'), '--model-index', str(args.model_index), '--slot', str(slot)], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            log.close()
        while any(p.poll() is None for p in processes):
            failed = [p.returncode for p in processes if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'Worker failure: {failed}')
            time.sleep(2)
        assert all(p.returncode == 0 for p in processes)
        subprocess.run([spec['verification_python'], str(ROOT / 'score.py'), '--model', model['tag']], check=True)
        subprocess.run([spec['verification_python'], str(ROOT / 'summarize.py')], check=True)
        write_json(output / 'complete.json', {'status': 'complete', 'responses': 1536, 'job_id': os.environ['SLURM_JOB_ID']})
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)


if __name__ == '__main__':
    main()
