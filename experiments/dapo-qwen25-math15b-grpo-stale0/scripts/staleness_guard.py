import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tomllib


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def config(root):
    return tomllib.loads((root / 'config/main.toml').read_text())


def fingerprint(root):
    return digest(root / 'source-manifest.json')


def static(root):
    pair = read(root / 'comparison.json')
    hashes = read(root / 'source-manifest.json')
    for rel, expected in hashes.items():
        require(digest(root / rel) == expected, f'Frozen source changed: {rel}')
    current = config(root)
    reference = tomllib.loads((root / 'baseline/main.toml').read_text())
    expected_text = (root / 'baseline/main.toml').read_text()
    for old, new in pair['path_replacements'].items():
        expected_text = expected_text.replace(old, new)
    expected = tomllib.loads(expected_text)
    require(reference['orchestrator']['max_off_policy_steps'] == 2, 'Reference cap must be 2')
    expected['orchestrator']['max_off_policy_steps'] = 0
    require(current == expected, 'Production settings differ beyond cap and run identity')
    require('resume' not in current, 'Production config must not embed a resume target')
    topology = current['deployment']
    require(topology['gpus_per_node'] == topology['num_train_gpus'] + topology['num_infer_gpus'], 'Unassigned GPUs')
    vllm = current['inference']['vllm']
    require(vllm['tensor_parallel_size'] * vllm['data_parallel_size'] == topology['num_infer_gpus'], 'Inference topology mismatch')
    for rel in ['python/ppo_loss.py', 'python/exact_math.py', 'config/data_exclusions.json']:
        require(digest(root / rel) == read(root / 'baseline/manifest.json')['input_hashes'][rel], f'Baseline algorithm or data changed: {rel}')
    spec = read(root / 'experiment.json')
    for key in ['model', 'model_revision', 'dataset', 'dataset_revision', 'seed', 'algorithm', 'clip_epsilon', 'kl_beta', 'group_size', 'train_batch_size']:
        require(spec[key] == read(root / 'baseline/experiment.json')[key], f'Baseline metadata changed: {key}')
    require(spec['checkpoint_target'] == current['output_dir'] + '/' + current['run']['name'], 'Checkpoint path mismatch')
    return pair


def materialize(root):
    pair = static(root)
    base = Path(pair['baseline_root'])
    original = read(root / 'baseline/manifest.json')
    require(digest(base / 'data/manifest.json') == digest(root / 'baseline/manifest.json'), 'Baseline manifest changed since capture')
    expected = dict(original)
    expected['input_hashes'] = read(root / 'source-manifest.json') | {'source-manifest.json': fingerprint(root)}
    expected['tokenizer'] = str(root / 'tokenizer')
    target = root / 'data/manifest.json'
    if target.exists():
        require(read(target) == expected, 'Refusing to replace an existing experiment fingerprint')
    else:
        require(not Path(pair['run_dir']).exists(), 'Refusing to materialize over existing run state')
        require(not Path(pair['observer_dir']).exists(), 'Refusing to materialize over existing observer state')
    for folder, hashes in [('data', original['hashes']), ('tokenizer', original['tokenizer_hashes'])]:
        for rel, sha in hashes.items():
            src, dst = base / folder / rel, root / folder / rel
            require(digest(src) == sha, f'Baseline asset changed: {folder}/{rel}')
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            require(digest(dst) == sha, f'Copied asset mismatch: {folder}/{rel}')
    if not target.exists():
        target.write_text(json.dumps(expected, indent=2, sort_keys=True) + '\n')


def devices(root):
    static(root)
    topology = config(root)['deployment']
    ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    require(len(ids) == topology['gpus_per_node'] and len(set(ids)) == len(ids), 'GPU allocation does not match the frozen topology')
    require(all(ids), 'Empty GPU identifier')
    split = topology['num_train_gpus']
    return ','.join(ids[split:] + ids[:split])


def audit(root, run, steps):
    static(root)
    cfg = read(run / 'configs/resolved/orchestrator.json')
    require(cfg['max_off_policy_steps'] == 0, 'Resolved runtime cap is not zero')
    by_step = {}
    with (run / 'metrics.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            if 'step' in row:
                by_step.setdefault(int(row['step']), {}).update(row)
    required_metrics = ['off_policy/mean', 'off_policy/max', 'off_policy/in_flight/mean', 'off_policy/in_flight/max', 'off_policy/in_queue/mean', 'off_policy/in_queue/max']
    counts = {}
    for step in range(1, steps + 1):
        metrics = by_step.get(step, {})
        for key in required_metrics:
            require(key in metrics and metrics[key] == 0, f'Step {step}: missing or nonzero {key}')
        require('optim/grad_norm' in metrics and math.isfinite(metrics['optim/grad_norm']), f'Step {step}: missing or nonfinite gradient')
        require('loss/mean' in metrics and math.isfinite(metrics['loss/mean']), f'Step {step}: missing or nonfinite loss')
        count = 0
        traces = run / f'rollouts/step_{step}/train/effective/traces.jsonl'
        with traces.open() as handle:
            for line in handle:
                episode = json.loads(line)
                policy = episode.get('run', {}).get('work', {}).get('policy')
                require(policy and policy.get('start') == step - 1 and policy.get('end') == step - 1, f'Step {step}: shipped rollout used the wrong policy version')
                count += len(episode['traces'])
        require(count > 0, f'Step {step}: no shipped traces')
        counts[str(step)] = count
    for rel in ['trainer/.metadata', 'orchestrator/progress.pt']:
        checkpoint = run / f'checkpoints/step_{steps}' / rel
        require(checkpoint.is_file() and checkpoint.stat().st_size > 0, f'Missing paired checkpoint: {rel}')
    return {'source_sha256': fingerprint(root), 'run_dir': str(run), 'steps': steps, 'effective_trace_counts': counts, 'status': 'passed'}


def gate(root):
    static(root)
    report = read(root / 'validation/smoke.json')
    require(report['source_sha256'] == fingerprint(root) and report['steps'] == 5 and report['status'] == 'passed', 'A matching five-update zero-staleness smoke test is required')
    require(audit(root, Path(report['run_dir']), 5) == report, 'Smoke evidence changed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['static', 'materialize', 'devices', 'audit', 'smoke', 'gate'])
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--steps', type=int, default=5)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    if args.action == 'static':
        static(root)
    elif args.action == 'materialize':
        materialize(root)
    elif args.action == 'devices':
        print(devices(root))
        return
    elif args.action == 'gate':
        gate(root)
    else:
        require(args.run_dir is not None, '--run-dir is required')
        report = audit(root, args.run_dir, args.steps)
        if args.action == 'smoke':
            require(args.steps == 5, 'Smoke acceptance requires five updates')
            (root / 'validation').mkdir(exist_ok=True)
            (root / 'validation/smoke.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
        return
    print(json.dumps({'action': args.action, 'experiment': root.name, 'status': 'passed'}))


if __name__ == '__main__':
    main()
