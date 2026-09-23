import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from allocation import inspect_allocation

PACKAGE = Path(__file__).resolve().parent
FROZEN_SHA256 = 'a0b2a53a0c73ff78f3257a1f730dccc327217275570ebc0cd72aff8086fb836c'


def verify_package():
    path = PACKAGE / 'source-manifest.json'
    manifest = json.loads(path.read_text())
    for name, sha in manifest['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Full-node wrapper changed: {name}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dispatcher(source):
    path = source / 'evaluation/launches/3b-math-node5-20260920/dispatch.py'
    spec = importlib.util.spec_from_file_location('frozen_3b_dispatcher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.verify_package() != FROZEN_SHA256:
        raise ValueError('Frozen evaluation implementation changed')
    return module


def probe_gpu(device):
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES=device, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = subprocess.run([sys.executable, str(PACKAGE / 'gpu_probe.py')], env=environment, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError({'gpu_uuid': device, 'stdout': result.stdout[-2000:], 'stderr': result.stderr[-3000:]})
    receipt = json.loads(result.stdout)
    if receipt.get('status') != 'passed' or receipt.get('gpu_uuid') != device or not receipt.get('bf16_backward'):
        raise ValueError('GPU probe receipt did not pass for the assigned device')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    for name in ('frozen-source', 'plan', 'data', 'prepared-root', 'results-root', 'tokenizer-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.workers_from_visible = True
    wrapper_sha = verify_package()
    frozen = load_dispatcher(args.frozen_source)
    entries, identity = frozen.prepared_entries(args, FROZEN_SHA256)
    if len(entries) != 140:
        raise ValueError('Expected all 140 original production cells')
    info = inspect_allocation()
    directory = args.results_root / '.full-node' / f"{info['job_id']}-{os.environ.get('SLURM_RESTART_COUNT', '0')}"
    directory.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'validating', 'wrapper_manifest_sha256': wrapper_sha, 'frozen_manifest_sha256': FROZEN_SHA256, 'allocation': info, 'gpu_checks': [], **identity}
    path = directory / 'validation.json'
    frozen.write_json(path, receipt)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
            futures = [pool.submit(probe_gpu, device) for device in info['worker_gpu_uuids']]
            for future in concurrent.futures.as_completed(futures):
                receipt['gpu_checks'].append(future.result())
                frozen.write_json(path, receipt)
        if {item['gpu_uuid'] for item in receipt['gpu_checks']} != set(info['worker_gpu_uuids']):
            raise ValueError('Every allocated GPU must pass BF16 validation')
        receipt['status'] = 'passed'
        frozen.write_json(path, receipt)
    except BaseException as error:
        receipt.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        frozen.write_json(path, receipt)
        raise
    info['full_node_wrapper'] = {'manifest_sha256': wrapper_sha, 'validation_path': str(path), 'validation_sha256': frozen.file_hash(path), 'bf16_gpus_passed': len(receipt['gpu_checks'])}
    frozen.allocation = lambda: info
    result = frozen.run(args, FROZEN_SHA256)
    print(json.dumps({'status': result['status'], 'cells': len(result.get('completed', []))}), flush=True)


if __name__ == '__main__':
    main()
