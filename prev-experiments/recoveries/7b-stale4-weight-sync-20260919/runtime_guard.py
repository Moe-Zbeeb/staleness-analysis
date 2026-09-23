import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path


BASE_COMMIT = 'ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1'
CLIENTS = 'src/prime_rl/orchestrator/clients.py'
ORIGINAL_CLIENTS_SHA256 = 'a2d117b78b34c638cc19f3a92e8ec166634a23120319a14ae51ea570afa9fcc7'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_source(root, upstream_root=None):
    root = Path(root).resolve()
    source = root / 'prime-rl-source'
    provenance = read(root / 'runtime-provenance.json')
    baseline = read(root / 'prime-rl-source-original.json')
    current = read(root / 'prime-rl-source-manifest.json')
    require(provenance['base_prime_commit'] == BASE_COMMIT, 'Runtime base commit changed')
    require(provenance['original_manifest_sha256'] == digest(root / 'prime-rl-source-original.json'), 'Original runtime manifest changed')
    require(provenance['runtime_manifest_sha256'] == digest(root / 'prime-rl-source-manifest.json'), 'Patched runtime manifest changed')
    require(set(current) == set(baseline), 'Runtime source added or removed upstream files')
    if upstream_root is not None:
        upstream_root = Path(upstream_root).resolve()
        for name, expected in baseline.items():
            path = (upstream_root / name).resolve()
            require(path.is_relative_to(upstream_root) and path.is_file() and digest(path) == expected, f'Live upstream source differs from the pinned baseline: {name}')
    changes = {name for name in current if current[name] != baseline[name]}
    require(changes == {CLIENTS}, f'Runtime source changes exceed the weight-update client: {changes}')
    actual = {str(path.relative_to(source)) for path in source.rglob('*') if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
    require(actual == set(current), 'Runtime source contains missing or unmanifested files')
    for name, expected in current.items():
        path = (source / name).resolve()
        require(path.is_relative_to(source) and digest(path) == expected, f'Runtime source file changed: {name}')
    original = root / 'upstream/clients.py'
    require(digest(original) == baseline[CLIENTS] == ORIGINAL_CLIENTS_SHA256, 'Pristine weight client differs from the deployed source')
    require(provenance['original_clients_sha256'] == ORIGINAL_CLIENTS_SHA256, 'Original client provenance changed')
    require(provenance['patched_clients_sha256'] == current[CLIENTS], 'Patched client provenance changed')
    require(provenance['helper_sha256'] == digest(root / 'weight_sync_recovery.py'), 'Weight recovery helper changed')
    helper_spec = importlib.util.spec_from_file_location('weight_sync_patch_validation', root / 'weight_sync_recovery.py')
    require(helper_spec is not None and helper_spec.loader is not None, 'Weight helper is unavailable')
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    require((source / CLIENTS).read_text() == helper.apply_source_patch(original.read_text()), 'Client source does not exactly match the declared bounded patch')
    require(provenance['read_timeout_seconds'] == 3600 and provenance['update_attempts'] == 1 and provenance['resume_only_after_confirmed_update'] is True, 'Weight recovery semantics changed')
    return provenance


def verify_imports(root, upstream_root):
    root = Path(root).resolve()
    provenance = verify_source(root, upstream_root)
    paths = {
        'prime_rl.orchestrator.clients': root / 'prime-rl-source/src/prime_rl/orchestrator/clients.py',
        'prime_rl.entrypoints.rl': root / 'prime-rl-source/src/prime_rl/entrypoints/rl.py',
        'prime_rl.inference.vllm.server': root / 'prime-rl-source/src/prime_rl/inference/vllm/server.py',
        'prime_rl.configs.rl': root / 'prime-rl-source/packages/prime-rl-configs/src/prime_rl/configs/rl.py',
        'weight_sync_recovery': root / 'weight_sync_recovery.py',
    }
    origins = {}
    for name, expected in paths.items():
        spec = importlib.util.find_spec(name)
        require(spec is not None and spec.origin is not None and Path(spec.origin).resolve() == expected, f'Runtime import escapes the isolated recovery: {name}')
        origins[name] = str(expected)
    clients = importlib.import_module('prime_rl.orchestrator.clients')
    helper = importlib.import_module('weight_sync_recovery')
    require(clients.update_weights is helper.update_weights, 'Weight-update client did not bind the recovery implementation')
    return {'status': 'passed', 'base_prime_commit': BASE_COMMIT, 'runtime_provenance_sha256': digest(root / 'runtime-provenance.json'), 'runtime_manifest_sha256': provenance['runtime_manifest_sha256'], 'module_origins': origins, 'upstream_root': str(Path(upstream_root).resolve()), 'upstream_original_manifest_sha256': provenance['original_manifest_sha256'], 'upstream_source_files_verified': len(read(root / 'prime-rl-source-original.json')), 'job_id': os.environ.get('SLURM_JOB_ID'), 'restart_count': os.environ.get('SLURM_RESTART_COUNT', '0')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recovery-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--upstream-root', type=Path, required=True)
    args = parser.parse_args()
    result = verify_imports(args.recovery_root, args.upstream_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        require(read(args.output) == result, 'Existing runtime proof differs')
    else:
        with args.output.open('x') as stream:
            stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
