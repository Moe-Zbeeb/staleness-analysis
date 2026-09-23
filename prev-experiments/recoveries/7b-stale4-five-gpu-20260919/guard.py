import argparse
import datetime
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
from pathlib import Path


SHARDS = [f'trainer/__{rank}_0.distcp' for rank in range(4)]
PAIRED_FILES = ['trainer/.metadata', 'orchestrator/progress.pt', *SHARDS]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def child(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f'Evidence path escapes its root: {relative}')
    return path


class Recovery:
    def __init__(self, recovery_root, experiment_root):
        self.root = recovery_root.resolve()
        self.experiment = experiment_root.resolve()
        self.spec_path = self.root / 'spec.json'
        self.spec = read(self.spec_path)
        selected = [(int(cap), item) for cap, item in self.spec['caps'].items() if Path(item['experiment_root']).resolve() == self.experiment]
        require(len(selected) == 1, 'Experiment must match exactly one recovery cap')
        self.cap, self.item = selected[0]
        require(self.cap == 4, 'Recovery requires cap 4')
        self.run = Path(self.item['run_dir']).resolve()
        self.evidence = child(self.root, self.item['evidence_dir'])
        self.proof = {}
        self.original_path = self.experiment / 'scripts/staleness_guard.py'
        require(digest(self.original_path) == self.item['guard_sha256'], 'Original guard hash mismatch')
        module_spec = importlib.util.spec_from_file_location(f'original_guard_cap_{self.cap}', self.original_path)
        self.original = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(self.original)

    def static(self):
        package_path = self.root / 'package-manifest.json'
        require(package_path.is_file(), 'Recovery package manifest is required')
        if package_path.exists():
            package = read(package_path)
            required_package = {'guard.py', 'health_probe.py', 'gpu_devices.py', 'gres-reviewed.conf', 'spec.json', 'train-stale4.sh', 'validate_plan.py'}
            for item in self.spec['caps'].values():
                required_package.update(str(Path(item['evidence_dir']) / name) for name in item['evidence_hashes'])
            require(isinstance(package, dict) and required_package <= set(package), 'Recovery package manifest is incomplete')
            for relative, sha in package.items():
                require(relative != 'package-manifest.json' and isinstance(sha, str), 'Invalid package hash entry')
                require(digest(child(self.root, relative)) == sha, f'Recovery package file changed: {relative}')
        require(digest(self.original_path) == self.item['guard_sha256'], 'Original guard changed')
        require(digest(self.experiment / 'source-manifest.json') == self.item['source_sha256'], 'Original source manifest changed')
        require(digest(self.experiment / 'data/manifest.json') == self.item['data_sha256'], 'Original data manifest changed')
        self.original.static(self.experiment)
        cfg = self.original.config(self.experiment)
        require(Path(cfg['output_dir']) / cfg['run']['name'] == self.run, 'Original run identity mismatch')
        require(cfg['deployment'] == {'type': 'single_node', 'gpus_per_node': 5, 'num_train_gpus': 4, 'num_infer_gpus': 1}, 'Unexpected original topology')
        require(cfg['inference']['vllm']['tensor_parallel_size'] == 1 and cfg['inference']['vllm']['data_parallel_size'] == 1, 'Unexpected original inference topology')
        require(cfg['orchestrator']['max_off_policy_steps'] == self.cap, 'Original cap mismatch')
        require(self.item['resume_step'] == 225, 'Initial recovery checkpoint changed')
        required_evidence = {'checkpoint/trainer/.metadata', 'checkpoint/orchestrator/progress.pt', 'checkpoint-files.json', *[f'original-resolved/{component}.json' for component in ['trainer', 'orchestrator', 'inference']]}
        require(required_evidence <= set(self.item['evidence_hashes']), 'Required initial evidence is not hash-bound')
        for relative, sha in self.item['evidence_hashes'].items():
            require(digest(child(self.evidence, relative)) == sha, f'Initial evidence changed: {relative}')
        require(digest(self.evidence / 'checkpoint/trainer/.metadata') == self.item['metadata_sha256'], 'Initial metadata evidence mismatch')
        require(digest(self.evidence / 'checkpoint/orchestrator/progress.pt') == self.item['progress_sha256'], 'Initial progress evidence mismatch')
        require((self.evidence / 'checkpoint/trainer/.metadata').stat().st_size > 0 and (self.evidence / 'checkpoint/orchestrator/progress.pt').stat().st_size > 0, 'Initial paired evidence is empty')
        sizes = read(self.evidence / 'checkpoint-files.json')
        require(all(isinstance(sizes.get(name), int) and sizes[name] > 0 for name in SHARDS), 'Initial four-shard evidence missing')
        return cfg

    def checkpoint(self, step):
        checkpoint = self.run / f'checkpoints/step_{step}'
        require(all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0 for name in PAIRED_FILES), f'Incomplete paired four-trainer checkpoint: {step}')
        require({str(path.relative_to(checkpoint)) for path in (checkpoint / 'trainer').glob('*.distcp')} == set(SHARDS), 'Checkpoint trainer world size is not four')
        files = {name: {'bytes': (checkpoint / name).stat().st_size, 'mtime_ns': (checkpoint / name).stat().st_mtime_ns} for name in PAIRED_FILES}
        return checkpoint, files

    def capture(self, step):
        self.static()
        require(isinstance(step, int) and self.item['resume_step'] <= step < 1000, 'Invalid capture resume step')
        job = os.environ.get('SLURM_JOB_ID', '')
        restart = os.environ.get('SLURM_RESTART_COUNT', '0')
        require(job.isdigit() and restart.isdigit(), 'Capture requires a Slurm job identity')
        destination = self.evidence / 'attempts' / f'job-{job}-restart-{restart}-step-{step}'
        record_path = destination / 'capture.json'
        if record_path.exists():
            record = self.validate_capture(record_path, step)
            checkpoint, current = self.checkpoint(step)
            require(all(current[name]['bytes'] == record['checkpoint_files'][name]['bytes'] for name in PAIRED_FILES), 'Current checkpoint differs from existing capture')
            require(digest(checkpoint / 'trainer/.metadata') == record['evidence_hashes']['checkpoint/trainer/.metadata'] and digest(checkpoint / 'orchestrator/progress.pt') == record['evidence_hashes']['checkpoint/orchestrator/progress.pt'], 'Current paired checkpoint hashes differ from existing capture')
            return {'action': 'capture', 'capture_path': str(record_path), 'capture_sha256': digest(record_path), 'resume_step': record['resume_step'], 'status': 'passed'}
        checkpoint, before = self.checkpoint(step)
        metadata_sha = digest(checkpoint / 'trainer/.metadata')
        progress_sha = digest(checkpoint / 'orchestrator/progress.pt')
        if step == self.item['resume_step']:
            require(metadata_sha == self.item['metadata_sha256'] and progress_sha == self.item['progress_sha256'], 'Initial checkpoint differs from immutable evidence')
            initial = read(self.evidence / 'checkpoint-files.json')
            require(all(before[name]['bytes'] == initial[name] for name in SHARDS), 'Initial checkpoint shard sizes changed')
        destination.mkdir(parents=True, exist_ok=False)
        for name in ['trainer/.metadata', 'orchestrator/progress.pt']:
            output = destination / 'checkpoint' / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checkpoint / name, output)
        for component in ['trainer', 'orchestrator', 'inference']:
            output = destination / 'prelaunch-resolved' / f'{component}.json'
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.run / 'configs/resolved' / f'{component}.json', output)
        hashes = {str(path.relative_to(destination)): digest(path) for path in destination.rglob('*') if path.is_file()}
        require(hashes['checkpoint/trainer/.metadata'] == metadata_sha and hashes['checkpoint/orchestrator/progress.pt'] == progress_sha, 'Copied checkpoint evidence differs from source')
        require(self.checkpoint(step)[1] == before and digest(checkpoint / 'trainer/.metadata') == metadata_sha and digest(checkpoint / 'orchestrator/progress.pt') == progress_sha, 'Checkpoint changed during capture')
        record = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'cap': self.cap, 'run_dir': str(self.run), 'experiment_root': str(self.experiment), 'resume_step': step, 'job_id': job, 'restart_count': int(restart), 'spec_sha256': digest(self.spec_path), 'guard_sha256': digest(Path(__file__).resolve()), 'original_guard_sha256': self.item['guard_sha256'], 'source_sha256': self.item['source_sha256'], 'data_sha256': self.item['data_sha256'], 'checkpoint_files': before, 'evidence_hashes': hashes}
        with record_path.open('x') as handle:
            handle.write(json.dumps(record, indent=2) + '\n')
        for path in destination.rglob('*'):
            if path.is_file():
                path.chmod(0o444)
        self.validate_capture(record_path, step)
        return {'action': 'capture', 'capture_path': str(record_path), 'capture_sha256': digest(record_path), 'resume_step': step, 'status': 'passed'}

    def validate_capture(self, path, step):
        record = read(path)
        for key, value in {'cap': self.cap, 'run_dir': str(self.run), 'experiment_root': str(self.experiment), 'resume_step': step, 'spec_sha256': digest(self.spec_path), 'guard_sha256': digest(Path(__file__).resolve()), 'original_guard_sha256': self.item['guard_sha256'], 'source_sha256': self.item['source_sha256'], 'data_sha256': self.item['data_sha256']}.items():
            require(record.get(key) == value, f'Capture provenance mismatch: {key}')
        require(record['job_id'].isdigit() and isinstance(record['restart_count'], int), 'Invalid captured Slurm identity')
        require(path.parent.name == f"job-{record['job_id']}-restart-{record['restart_count']}-step-{step}", 'Capture path identity mismatch')
        require(set(record['checkpoint_files']) == set(PAIRED_FILES), 'Capture is missing checkpoint files')
        require(all(entry['bytes'] > 0 for entry in record['checkpoint_files'].values()), 'Capture contains empty files')
        required = {'checkpoint/trainer/.metadata', 'checkpoint/orchestrator/progress.pt', *[f'prelaunch-resolved/{component}.json' for component in ['trainer', 'orchestrator', 'inference']]}
        require(required == set(record['evidence_hashes']), 'Capture evidence coverage mismatch')
        for relative, sha in record['evidence_hashes'].items():
            require(digest(child(path.parent, relative)) == sha, f'Captured evidence changed: {relative}')
        if step == self.item['resume_step']:
            require(record['evidence_hashes']['checkpoint/trainer/.metadata'] == self.item['metadata_sha256'], 'Captured initial metadata mismatch')
            require(record['evidence_hashes']['checkpoint/orchestrator/progress.pt'] == self.item['progress_sha256'], 'Captured initial progress mismatch')
        return record

    def resume_evidence(self, root, run, step, steps):
        require(root == self.experiment and run == self.run, 'Resume audit changed run identity')
        require(self.item['resume_step'] <= step < steps, 'Resume is outside recovery lineage')
        metadata = run / f'checkpoints/step_{step}/trainer/.metadata'
        if metadata.is_file():
            self.checkpoint(step)
            if step == self.item['resume_step']:
                require(digest(metadata) == self.item['metadata_sha256'], 'Retained initial checkpoint metadata changed')
                require(digest(run / f'checkpoints/step_{step}/orchestrator/progress.pt') == self.item['progress_sha256'], 'Retained initial checkpoint progress changed')
            return
        require(not metadata.parent.parent.exists(), 'Resume checkpoint exists but is incomplete')
        retention = self.original.config(root)['ckpt']
        require(retention == {'interval': 25, 'keep_last': 4, 'keep_interval': 100} and step % retention['keep_interval'] != 0, 'Missing checkpoint is not explained by retention')
        newer = []
        for path in (run / 'checkpoints').glob('step_*'):
            suffix = path.name.removeprefix('step_')
            if suffix.isdigit() and int(suffix) > step and all((path / name).is_file() and (path / name).stat().st_size > 0 for name in PAIRED_FILES):
                newer.append(int(suffix))
        require(len(newer) >= retention['keep_last'] and steps in newer, 'Insufficient newer complete checkpoints')
        captures = []
        for path in sorted((self.evidence / 'attempts').glob(f'job-*-restart-*-step-{step}/capture.json')):
            self.validate_capture(path, step)
            captures.append({'path': str(path), 'sha256': digest(path)})
        require(captures, 'Missing immutable prelaunch capture for pruned resume checkpoint')
        attempts = [path for path in (run / 'logs').glob('attempt_*') if path.name.removeprefix('attempt_').isdigit()]
        require(attempts, 'Missing resumed attempt logs')
        latest = max(attempts, key=lambda path: int(path.name.removeprefix('attempt_')))
        logs = []
        for component in ['trainer', 'orchestrator']:
            path = latest / f'{component}.log'
            text = re.sub(r'\x1b\[[0-9;]*m', '', path.read_text())
            pattern = rf'Resuming from step {step} \(' if component == 'trainer' else rf'Resuming from step {step}(?:\s|$)'
            lines = [line for line in text.splitlines() if re.search(pattern, line)]
            require(lines, f'Missing successful {component} resume evidence in latest attempt')
            require((re.search(rf'SUCCESS Step {steps}\s*\|', text) is not None) if component == 'trainer' else f'Saving final checkpoint at step {steps}' in text, f'Missing {component} completion evidence')
            logs.append({'path': str(path), 'sha256': digest(path), 'resume_lines': lines})
        self.proof[step] = {'resume_step': step, 'retention_policy': retention, 'newer_paired_checkpoints': sorted(newer), 'prelaunch_captures': captures, 'successful_resume_logs': logs}

    def audit(self, run, steps):
        self.static()
        require(run.resolve() == self.run and steps == 1000, 'Recovery production audit requires original run at step 1000')
        original_runtime = self.original.runtime_parity
        source = inspect.getsource(original_runtime)
        checkpoint_line = 'require((run / f"checkpoints/step_{resume[\'step\']}/trainer/.metadata").is_file(), \'Resume checkpoint missing\')'
        require(source.count(checkpoint_line) == 1, 'Original runtime assertion structure changed')
        adapted = source.replace(checkpoint_line, "recovery_resume_evidence(root, run, resume['step'], steps)")
        namespace = dict(self.original.__dict__)
        namespace.update(recovery_resume_evidence=self.resume_evidence)
        exec(compile(adapted, str(self.original_path), 'exec'), namespace)
        self.original.runtime_parity = namespace['runtime_parity']
        try:
            report = self.original.audit(self.experiment, self.run, steps)
        finally:
            self.original.runtime_parity = original_runtime
        self.static()
        require(report['status'] == 'passed' and report['cap'] == self.cap, 'Unexpected original audit outcome')
        report.update(recovery_spec_sha256=digest(self.spec_path), recovery_guard_sha256=digest(Path(__file__).resolve()), original_guard_sha256=self.item['guard_sha256'], training_manifest_sha256=self.item['data_sha256'], initial_resume_step=self.item['resume_step'], execution_segment={'trainer_gpus': 4, 'inference_gpus': 1, 'inference_tensor_parallel_size': 1, 'inference_data_parallel_size': 1}, supplemental_resume_evidence=list(self.proof.values()), frozen_experiment_files_modified=False)
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['static', 'gate', 'devices', 'capture', 'audit'])
    parser.add_argument('--recovery-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--resume-step', type=int)
    parser.add_argument('--steps', type=int, default=1000)
    args = parser.parse_args()
    recovery = Recovery(args.recovery_root, args.experiment_root)
    recovery.static()
    if args.action == 'gate':
        recovery.original.gate(recovery.experiment)
    elif args.action == 'devices':
        ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        require(len(ids) == 5 and len(set(ids)) == 5 and all(value.isdigit() for value in ids), 'Allocation must expose exactly five distinct numeric CUDA identifiers')
        print(','.join(ids[4:] + ids[:4]))
        return
    elif args.action == 'capture':
        require(args.resume_step is not None, 'Capture requires --resume-step')
        print(json.dumps(recovery.capture(args.resume_step)))
        return
    elif args.action == 'audit':
        require(args.run_dir is not None, 'Audit requires --run-dir')
        job = os.environ.get('SLURM_JOB_ID', '')
        require(job.isdigit(), 'Production completion audit requires a Slurm job identity')
        report = recovery.audit(args.run_dir, args.steps)
        destination = recovery.root / 'validation' / f'completion-stale{recovery.cap}-job{job}.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(destination)
        print(json.dumps(report))
        return
    print(json.dumps({'action': args.action, 'experiment': recovery.experiment.name, 'cap': recovery.cap, 'status': 'passed'}))


if __name__ == '__main__':
    main()
