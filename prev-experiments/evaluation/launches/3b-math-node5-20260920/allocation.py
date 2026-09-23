import csv
import hashlib
import io
import os
import socket
import subprocess
from pathlib import Path

from native_devices import inspect_assignment, require


def inspect_allocation():
    job = os.environ.get('SLURM_JOB_ID', '')
    require(job.isdigit() and os.environ.get('SLURM_GPUS_ON_NODE') == '3', 'Requires three GPUs in a Slurm job')
    path = Path('/usr/local/etc/gres.conf')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    require(sha == '1a5c2624ff52be8400be632e1aed6f61b75399446b91d9eb249d19ac38307732', 'GRES configuration changed')
    visible, devices = inspect_assignment()
    selected = ['GPU-' + item['uuid'] for item in devices]
    selector = '--id=' + ','.join(selected)
    args = ['nvidia-smi', selector, '--query-gpu=uuid,name,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits']
    output = subprocess.run(args, check=True, capture_output=True, text=True, timeout=30).stdout
    inventory = []
    for row in csv.reader(io.StringIO(output)):
        require(len(row) == 5, 'Unexpected NVIDIA inventory schema')
        uid, name, total, free, used = [value.strip() for value in row]
        total, free, used = int(total), int(free), int(used)
        require(uid in selected and 'A100' in name and 37 <= total / 1024 < 85, 'Unexpected allocated GPU hardware')
        require(free >= total * 0.85, f'Insufficient free VRAM on allocated GPU {uid}')
        inventory.append({'uuid': uid, 'name': name, 'total_memory_mib': total, 'free_memory_mib': free, 'used_memory_mib': used})
    require(len(inventory) == 3 and {item['uuid'] for item in inventory} == set(selected), 'Incomplete or duplicate allocated GPU inventory')
    require(len({(item['name'], item['total_memory_mib']) for item in inventory}) == 1, 'All three workers must have identical GPU hardware')
    args = ['nvidia-smi', selector, '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits']
    output = subprocess.run(args, check=True, capture_output=True, text=True, timeout=30).stdout
    processes = [row for row in csv.reader(io.StringIO(output)) if row and row != ['No running processes found']]
    require(not processes, f'Allocated GPUs already have compute processes: {processes}')
    memory = {key: int(value.split()[0]) * 1024 for key, value in (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines()) if key in ('MemTotal', 'MemAvailable', 'MemFree', 'Cached')}
    require(memory['MemAvailable'] >= 72 * 1024**3, 'Less than 72 GiB host RAM available for three workers')
    return {'job_id': job, 'host': socket.gethostname(), 'gpu_count': 3, 'worker_gpu_uuids': selected, 'inventory': inventory, 'allocated_devices': devices, 'original_cuda_visible_devices': visible, 'slurm_job_gpus': os.environ['SLURM_JOB_GPUS'], 'cuda_device_order': os.environ['CUDA_DEVICE_ORDER'], 'gres_config_sha256': sha, 'memory_bytes': memory, 'preexisting_compute_processes': processes, 'assignment_basis': 'Original Slurm CUDA_VISIBLE_DEVICES, resolved through CUDA with PCI_BUS_ID ordering'}
