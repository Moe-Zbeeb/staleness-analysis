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
    expected['orchestrator']['max_off_policy_steps'] = pair['treatment_cap']
    for key, value in pair['allowed_setting_changes'].items():
        parts = key.split('.')
        cursor = expected
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value
    require(current == expected, 'Production settings differ beyond declared cap, topology, and run identity')
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


def valid_policy(policy, step, cap):
    if not isinstance(policy, dict):
        return False
    start, end = policy.get('start'), policy.get('end')
    return isinstance(start, int) and isinstance(end, int) and 0 <= (step - 1) - start <= cap and 0 <= start <= end <= step - 1


def runtime_parity(root, run, steps):
    import sys
    sys.path.insert(0, str(root / 'python'))
    from prime_rl.configs.rl import RLConfig
    from prime_rl.utils.config import cli, dump_resolved_config

    argv = sys.argv
    try:
        sys.argv = ['rl', '@', str(root / 'config/main.toml')]
        if steps == 25:
            sys.argv += ['@', str(root / 'config/smoke.toml')]
        expected = cli(RLConfig)
    finally:
        sys.argv = argv
    for component in ['trainer', 'orchestrator', 'inference']:
        wanted = dump_resolved_config(getattr(expected, component))
        actual = read(run / 'configs/resolved' / f'{component}.json')
        if component == 'inference':
            for key in ['deployment', 'slurm', 'output_dir', 'dry_run']:
                wanted.pop(key, None)
                actual.pop(key, None)
            model_key = wanted['vllm']['model']
            require(Path(actual['vllm']['model']).name == Path(model_key).name, 'Inference model identity changed')
            actual['vllm']['model'] = model_key
        else:
            model_key = wanted['model']['name']
            require(Path(actual['model']['name']).name == Path(model_key).name, f'{component} model identity changed')
            actual['model']['name'] = model_key
            actual['output_dir'] = wanted['output_dir']
            resume = actual.get('resume')
            if resume:
                require(component in ['trainer', 'orchestrator'] and resume.get('dir') is None and isinstance(resume.get('step'), int) and 0 < resume['step'] < steps, 'Invalid same-arm resume')
                require((run / f"checkpoints/step_{resume['step']}/trainer/.metadata").is_file(), 'Resume checkpoint missing')
            actual['resume'] = wanted.get('resume')
        if component == 'orchestrator':
            for key in ['HOME', 'UV_CACHE_DIR']:
                actual.get('env_vars', {}).pop(key, None)
                wanted.get('env_vars', {}).pop(key, None)
        require(actual == wanted, f'Resolved {component} differs from frozen config: {differences(wanted, actual)}')


def differences(expected, actual, prefix=''):
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(expected.keys() | actual.keys()):
            result.extend(differences(expected.get(key), actual.get(key), prefix + '.' + key))
        return result
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        return [item for i, (left, right) in enumerate(zip(expected, actual)) for item in differences(left, right, prefix + f'[{i}]')]
    return [] if expected == actual else [{'path': prefix, 'expected': expected, 'actual': actual}]


def final_evaluations(run, steps, metrics):
    expected = {'math500-smoke': 4} if steps == 25 else {'math500-pass1': 500, 'amc23-pass1': 40, 'aime24-pass1': 30, 'aime25-pass1': 30, 'minerva-pass1': 272, 'olympiadbench-pass1': 675, 'aime24-sampled': 240, 'aime25-sampled': 240, 'aime26-sampled': 240}
    counts = {}
    trace_ids = set()
    with (run / f'rollouts/step_{steps}/eval/all/traces.jsonl').open() as handle:
        for line in handle:
            episode = json.loads(line)
            policy = episode.get('run', {}).get('work', {}).get('policy')
            require(policy == {'start': steps, 'end': steps}, 'Final evaluation spans a different policy version')
            name = episode['env']['name']
            for trace in episode['traces']:
                require(trace['id'] not in trace_ids, 'Duplicate final evaluation trace')
                trace_ids.add(trace['id'])
                counts[name] = counts.get(name, 0) + 1
    require(counts == expected, f'Final evaluation coverage mismatch: {counts}')
    for name in expected:
        require(metrics.get(f'eval/{name}/all/has_error/mean') == 0, f'Final evaluation error or missing metric: {name}')
        score = metrics.get(f'eval/{name}/all/agent/reward/mean')
        require(score is not None and math.isfinite(score), f'Missing final evaluation score: {name}')
    return counts


def audit(root, run, steps):
    static(root)
    cfg = read(run / 'configs/resolved/orchestrator.json')
    cap = read(root / 'comparison.json')['treatment_cap']
    require(cfg['max_off_policy_steps'] == cap, 'Resolved runtime cap mismatch')
    runtime_parity(root, run, steps)
    by_step = {}
    with (run / 'metrics.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('step') is not None:
                by_step.setdefault(int(row['step']), {}).update(row)
    required_metrics = ['off_policy/mean', 'off_policy/max', 'off_policy/in_flight/mean', 'off_policy/in_flight/max', 'off_policy/in_queue/mean', 'off_policy/in_queue/max']
    counts = {}
    ages = {}
    for step in range(1, steps + 1):
        metrics = by_step.get(step, {})
        for key in required_metrics:
            require(key in metrics and math.isfinite(metrics[key]) and 0 <= metrics[key] <= cap, f'Step {step}: missing or out-of-bounds {key}')
        require('optim/grad_norm' in metrics and math.isfinite(metrics['optim/grad_norm']), f'Step {step}: missing or nonfinite gradient')
        require('loss/mean' in metrics and math.isfinite(metrics['loss/mean']), f'Step {step}: missing or nonfinite loss')
        count = 0
        traces = run / f'rollouts/step_{step}/train/effective/traces.jsonl'
        with traces.open() as handle:
            for line in handle:
                episode = json.loads(line)
                policy = episode.get('run', {}).get('work', {}).get('policy')
                require(valid_policy(policy, step, cap), f'Step {step}: shipped rollout violates the policy-age bound')
                count += len(episode['traces'])
                age = str((step - 1) - policy['start'])
                ages[age] = ages.get(age, 0) + len(episode['traces'])
        require(count > 0, f'Step {step}: no shipped traces')
        counts[str(step)] = count
    for rel in ['trainer/.metadata', 'orchestrator/progress.pt']:
        checkpoint = run / f'checkpoints/step_{steps}' / rel
        require(checkpoint.is_file() and checkpoint.stat().st_size > 0, f'Missing paired checkpoint: {rel}')
    eval_counts = final_evaluations(run, steps, by_step[steps])
    return {'source_sha256': fingerprint(root), 'run_dir': str(run), 'steps': steps, 'effective_trace_counts': counts, 'age_histogram': ages, 'final_evaluation_counts': eval_counts, 'cap': cap, 'status': 'passed'}


def gate(root):
    static(root)
    report = read(root / 'validation/smoke.json')
    require(report['source_sha256'] == fingerprint(root) and report['steps'] == 25 and report['status'] == 'passed', 'A matching 25-update cap-4 smoke test is required')
    require(audit(root, Path(report['run_dir']), 25) == report, 'Smoke evidence changed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['static', 'materialize', 'devices', 'audit', 'smoke', 'gate'])
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--steps', type=int, default=25)
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
            require(args.steps == 25, 'Smoke acceptance requires 25 updates')
            (root / 'validation').mkdir(exist_ok=True)
            (root / 'validation/smoke.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
        return
    print(json.dumps({'action': args.action, 'experiment': root.name, 'status': 'passed'}))


if __name__ == '__main__':
    main()
