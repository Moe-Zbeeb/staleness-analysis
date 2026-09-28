import hashlib
import json
import subprocess
import sys
from pathlib import Path

from deepseek_study.config import StudyConfig
from deepseek_study.runtime.build import resolve

root = Path('/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study')
control = root / 'launches/deepseek7b-dapo17k-b32-hp-20260929-v1'
release = control / 'release'
receipt = json.loads((control / 'preparation-complete.json').read_text())
assert receipt['included_rows'] == 17005 and receipt['question_order_matches_15b']
base = json.loads((control / 'preparation-study.json').read_text())
allowed = {'lag', 'trainer_gpus', 'inference_gpus', 'inference_port', 'output_dir', 'historical_rollouts', 'local_inference_pool'}
result = {}
for directory, manifest in [(release, 'PACKAGE_SHA256.json')] + [(control / role, 'CONTROL_SHA256.json') for role in ('learner', 'worker', 'onpolicy')]:
    for name, expected in json.loads((directory / manifest).read_text()).items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, str(directory / name)
for role, lag, train, infer, allocated in [('learner', 256, 4, 4, 8), ('worker', 256, 4, 4, 4), ('onpolicy', 0, 3, 3, 6)]:
    raw = json.loads((control / role / 'study.json').read_text())
    for key, value in base.items():
        if key not in allowed:
            assert raw[key] == value, (role, key)
    study = StudyConfig.model_validate(raw)
    spec = json.loads((control / role / 'storage-spec.json').read_text())
    assert (study.lag, study.trainer_gpus, study.inference_gpus) == (lag, train, infer)
    assert study.response_batch_size == 32 and study.sequence_length == 8192
    assert spec['allocated_gpus'] == allocated and spec['minimum_gpu_bytes'] == 79000000000
    assert (infer if role == 'worker' else train + infer) == allocated
    assert not Path(spec['backup']).exists()
    if role != 'worker':
        config = resolve(study)
        assert config.weight_broadcast.type == 'filesystem'
        assert len(config.orchestrator.model.client.admin_base_url) == infer
        assert config.deployment.num_train_gpus == train
        (control / role / 'resolved-rl.json').write_text(config.model_dump_json(indent=2) + '\n')
    result[role] = {'lag': lag, 'allocated_gpus': allocated, 'trainer_gpus': 0 if role == 'worker' else train, 'inference_gpus': infer, 'gpu_execution_verified': False}
tests = ['tests/test_queue.py', 'tests/test_algorithm.py', 'tests/test_local_inference_pool_config.py', 'tests/test_inference_pool.py']
subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *tests], cwd=release, check=True)
(control / 'validation-complete.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
