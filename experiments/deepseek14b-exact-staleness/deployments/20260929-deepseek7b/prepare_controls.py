import hashlib
import json
import shutil
from pathlib import Path

root = Path('/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study')
control = root / 'launches/deepseek7b-dapo17k-b32-hp-20260929-v1'
parent = root / 'launches/deepseek15b-dapo17k-async-b32-hp-20260928-v1'
receipt = json.loads((control / 'preparation-complete.json').read_text())
assert receipt['included_rows'] == 17005 and receipt['excluded_rows'] == 0
assert receipt['question_order_matches_15b'] is True
base = json.loads((control / 'preparation-study.json').read_text())
release = control / 'release'
roles = [('learner', 256, 4, 4, 8, 21000), ('worker', 256, 4, 4, 4, 31000), ('onpolicy', 0, 3, 3, 6, 22000)]
for role, lag, trainer, inference, allocated, port in roles:
    destination = control / role
    destination.mkdir(exist_ok=False)
    name = f'deepseek7b-dapo17k-6k-exact{lag}-b32-seed42-v1'
    if role == 'worker':
        name += '-historical'
    study = {**base, 'lag': lag, 'trainer_gpus': trainer, 'inference_gpus': inference, 'inference_port': port, 'output_dir': str(root / 'outputs' / name)}
    if lag == 0:
        study.update(historical_rollouts=None, local_inference_pool=True)
    assert not Path(study['output_dir']).exists()
    (destination / 'study.json').write_text(json.dumps(study, indent=2) + '\n')
    spec = {'baseline_study': str(destination / 'study.json'), 'release': str(release), 'shared_prime': str(root / 'vendor/prime-rl'), 'workspace': '/tmp/mohamadzbib-staleness-storage-v2/' + name, 'runtime': '/tmp/mohamadzbib-staleness-runtime/deepseek7b-dapo-b32-v1-' + role, 'run_name': name, 'backup': str(root / 'outputs' / name), 'metrics': '/mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics/' + name, 'minimum_gpu_bytes': 79000000000, 'allocated_gpus': allocated, 'install_router': True}
    (destination / 'storage-spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    for path in (release / 'scripts').glob('*.py'):
        shutil.copy2(path, destination / path.name)
    template_role = 'worker' if role == 'worker' else 'learner'
    template = parent / template_role
    wheel = 'vllm_router-0.2.0-cp38-abi3-manylinux_2_28_x86_64.whl'
    shutil.copy2(template / wheel, destination / wheel)
    text = (template / 'job.sh').read_text().replace(str(template), str(destination))
    previous = json.loads((template / 'storage-spec.json').read_text())
    text = text.replace(previous['workspace'], spec['workspace']).replace(previous['run_name'], name)
    text = text.replace('export PYTHONUNBUFFERED=1', 'export CUDA_DEVICE_ORDER=PCI_BUS_ID\nexport PYTHONUNBUFFERED=1')
    (destination / 'job.sh').write_text(text)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in destination.iterdir() if path.is_file()}
    (destination / 'CONTROL_SHA256.json').write_text(json.dumps(hashes, indent=2) + '\n')
print(json.dumps({'control': str(control), 'roles': roles, 'preparation': receipt}, indent=2))
