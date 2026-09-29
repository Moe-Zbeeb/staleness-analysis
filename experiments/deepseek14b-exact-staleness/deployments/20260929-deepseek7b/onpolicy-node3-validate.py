import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from deepseek_study.config import StudyConfig
from deepseek_study.runtime.build import resolve

root = Path('/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek7b-dapo17k-b32-hp-20260929-v1')
control = root / 'onpolicy'
release = root / 'release'
previous = json.loads((root / 'onpolicy-before-node3/study.json').read_text())
current = json.loads((control / 'study.json').read_text())
assert {k: v for k, v in current.items() if k not in {'trainer_gpus', 'inference_gpus'}} == {k: v for k, v in previous.items() if k not in {'trainer_gpus', 'inference_gpus'}}
for directory, name in ((release, 'PACKAGE_SHA256.json'), (control, 'CONTROL_SHA256.json')):
    for relative, expected in json.loads((directory / name).read_text()).items():
        assert hashlib.sha256((directory / relative).read_bytes()).hexdigest() == expected, str(directory / relative)
study = StudyConfig.model_validate(current)
spec = json.loads((control / 'storage-spec.json').read_text())
assert (study.lag, study.trainer_gpus, study.inference_gpus, study.inference_tensor_parallel) == (0, 6, 1, 1)
assert study.local_inference_pool and study.response_batch_size == 32 and study.sequence_length == 8192
assert spec['allocated_gpus'] == 7 and spec['minimum_gpu_bytes'] == 39000000000
assert not Path(spec['backup']).exists()
config = resolve(study)
assert config.weight_broadcast.type == 'filesystem'
assert len(config.orchestrator.model.client.admin_base_url) == 1
assert config.deployment.num_train_gpus == 6 and config.deployment.num_infer_gpus == 1
(control / 'resolved-rl.json').write_text(config.model_dump_json(indent=2) + '\n')
result = {'validated_at': datetime.now(timezone.utc).isoformat(), 'job_id': 2145690, 'node': 'deep-chungus-3', 'trainer_gpus': 6, 'inference_gpus': 1, 'allocated_gpus': 7, 'lag': 0, 'responses_per_update': 32, 'context_tokens': 8192, 'scientific_configuration_unchanged': True, 'frozen_release_unchanged': True, 'gpu_execution_verified': False, 'control_sha256': hashlib.sha256((control / 'CONTROL_SHA256.json').read_bytes()).hexdigest()}
(root / 'onpolicy-node3-validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
