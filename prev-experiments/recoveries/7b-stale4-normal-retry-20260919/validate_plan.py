import datetime
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import cli, dump_resolved_config

from guard import Recovery, digest, require


root = Path(__file__).resolve().parent
spec = json.loads((root / 'spec.json').read_text())
require(set(spec['caps']) == {'4'}, 'Preflight requires exactly the 7B cap-4 recovery')
item = spec['caps']['4']
experiment = Path(item['experiment_root'])
recovery = Recovery(root, experiment)
recovery.static()
recovery.original.gate(experiment)
sys.path.insert(0, str(experiment / 'python'))
original_argv = sys.argv
try:
    sys.argv = ['rl', '@', str(experiment / 'config/main.toml')]
    frozen = cli(RLConfig)
    sys.argv = ['rl', '@', str(experiment / 'config/main.toml'), '@', str(root / 'overlay.toml')]
    current = cli(RLConfig)
finally:
    sys.argv = original_argv
differences = recovery.original.differences(dump_resolved_config(frozen), dump_resolved_config(current))
require(frozen.orchestrator.ckpt.wait_for_weights_timeout is None, 'Pinned original startup wait configuration changed')
expected = {'.orchestrator.ckpt.wait_for_weights_timeout': (None, 7200)}
actual = {entry['path']: (entry['expected'], entry['actual']) for entry in differences}
require(actual == expected, f'Unexpected resolved configuration changes: {differences}')
require(current.orchestrator.ckpt.wait_for_weights_timeout == 7200, 'Startup weight wait timeout changed')
require(current.deployment.gpus_per_node == 5, 'Total GPU count changed')
require(current.deployment.num_train_gpus == 4, 'Trainer world size changed')
require(current.deployment.num_infer_gpus == 1, 'Inference GPU count changed')
require(current.inference.vllm.tensor_parallel_size == 1, 'Inference tensor parallelism changed')
require(current.inference.vllm.data_parallel_size == 1, 'Inference data parallelism changed')
require(current.inference.vllm.api_server_count == 1, 'Inference API count changed')
require(current.trainer.weight_broadcast.inference_world_size == 1, 'Trainer broadcast world size changed')
require(current.orchestrator.weight_broadcast.inference_world_size == 1, 'Orchestrator broadcast world size changed')
require(current.orchestrator.max_off_policy_steps == 4, 'Staleness bound changed')
require(current.trainer.ckpt.skip_optimizer is False and current.trainer.ckpt.skip_progress is False and current.trainer.ckpt.skip_scheduler is False, 'Resume skips training state')
checkpoint, files = recovery.checkpoint(225)
require(digest(checkpoint / 'trainer/.metadata') == item['metadata_sha256'], 'Trainer metadata differs from captured evidence')
require(digest(checkpoint / 'orchestrator/progress.pt') == item['progress_sha256'], 'Orchestrator state differs from captured evidence')
payload = {'app': {'progress': {'step': 0}}}
reader = dcp.FileSystemReader(checkpoint / 'trainer')
metadata = reader.read_metadata()
dcp.load(payload, storage_reader=reader, no_dist=True)
require(payload['app']['progress']['step'] == 225, f'Unexpected trainer checkpoint step: {payload}')
keys = list(metadata.state_dict_metadata)
for prefix in ['app.model.', 'app.optimizers.', 'app.scheduler.', 'app.progress.']:
    require(any(key.startswith(prefix) for key in keys), f'Missing checkpoint state: {prefix}')
state = torch.load(checkpoint / 'orchestrator/progress.pt', map_location='cpu', weights_only=False)
progress = asdict(state['progress'])
require(progress['step'] == 226, f'Unexpected orchestrator next step: {progress}')
require('dapo-math' in state['train_source']['envs'], 'Training-source checkpoint state is missing')
device_tests = {}
for visible, wanted in [
    ('0,2,3,6,7', '7,0,2,3,6'),
    ('1,2,3,4,5', '5,1,2,3,4'),
]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible)
    result = subprocess.run([sys.executable, str(root / 'guard.py'), 'devices', '--experiment-root', str(experiment)], env=env, text=True, capture_output=True, check=True)
    require(result.stdout.strip() == wanted, f'Incorrect CUDA mapping: {result.stdout}')
    device_tests[visible] = result.stdout.strip()
for visible in ['0,1,2,3', '0,0,1,2,3', '0,,2,3,4', '0,1,2,3,4,5', 'GPU-a,GPU-b,GPU-c,GPU-d,GPU-e']:
    result = subprocess.run([sys.executable, str(root / 'guard.py'), 'devices', '--experiment-root', str(experiment)], env=dict(os.environ, CUDA_VISIBLE_DEVICES=visible), text=True, capture_output=True)
    require(result.returncode != 0 and 'exactly five distinct numeric CUDA identifiers' in result.stderr, f'Invalid allocation was accepted: {visible}')
report = {
    'status': 'passed',
    'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'job_id': os.environ.get('SLURM_JOB_ID'),
    'model': 'Qwen2.5-Math-7B',
    'cap': 4,
    'resume_step': 225,
    'trainer_saved_step': payload['app']['progress']['step'],
    'orchestrator_next_step': progress['step'],
    'trainer_shards': sorted(path.name for path in (checkpoint / 'trainer').glob('*.distcp')),
    'checkpoint_files': files,
    'changes': differences,
    'device_mapping_tests': device_tests,
    'source_sha256': item['source_sha256'],
    'data_sha256': item['data_sha256'],
    'tensor_metadata_entries': len(keys),
    'package_manifest_sha256': digest(root / 'package-manifest.json'),
}
recovery.static()
(root / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
