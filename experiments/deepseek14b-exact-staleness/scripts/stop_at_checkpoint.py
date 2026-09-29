import argparse
import hashlib
import json
import os
import re
import select
import signal
import stat
import time
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def immutable_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def owned_json(path):
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
        raise ValueError(f'Expected an owned regular JSON file: {path}')
    return json.loads(path.read_text())


def ancestry_record(pid, proc_root):
    process = proc_root / str(pid)
    fields = (process / 'stat').read_text().rsplit(')', 1)[1].split()
    if fields[0] in {'Z', 'X'}:
        raise ValueError('Launcher ancestry contains an exited process')
    return {
        'pid': pid, 'ppid': int(fields[1]), 'uid': process.stat().st_uid, 'start_time': fields[19],
    }


def slurm_ancestor(pid, job_id, proc_root):
    chain = []
    seen = set()
    try:
        for _ in range(32):
            if pid <= 1 or pid in seen:
                raise ValueError('Launcher has no stable job-specific Slurm ancestor')
            seen.add(pid)
            record = ancestry_record(pid, proc_root)
            chain.append(record)
            if record['uid'] == 0:
                process = proc_root / str(pid)
                command = (process / 'cmdline').read_bytes().decode().replace('\0', ' ').strip()
                if (
                    (process / 'comm').read_text().strip() != 'slurmstepd'
                    or command != f'slurmstepd: [{job_id}.batch]'
                ):
                    raise ValueError('Root-owned ancestor is not the intended Slurm batch step')
                if any(ancestry_record(item['pid'], proc_root) != item for item in chain):
                    raise ValueError('Launcher ancestry changed during validation')
                return {'method': 'root_owned_slurmstepd_ancestor', 'ancestor': record, 'command': command}
            if record['uid'] != os.geteuid():
                raise ValueError('Launcher ancestry crosses another user')
            pid = record['ppid']
    except (OSError, IndexError, UnicodeError) as error:
        raise ValueError('Launcher Slurm ancestry could not be verified') from error
    raise ValueError('Launcher Slurm ancestry exceeds the bounded depth')


def process_record(pid, start_time, study, job_id, proc_root=Path('/proc')):
    process = proc_root / str(pid)
    if process.stat().st_uid != os.geteuid():
        raise ValueError('Launcher belongs to another user')
    fields = (process / 'stat').read_text().rsplit(')', 1)[1].split()
    if fields[19] != str(start_time) or fields[0] in {'Z', 'X'}:
        raise ValueError('Launcher start time changed or process exited')
    statuses = dict(line.split(':', 1) for line in (process / 'status').read_text().splitlines() if ':' in line)
    if any(int(value) != os.geteuid() for value in statuses['Uid'].split()):
        raise ValueError('Launcher user identities differ from the current user')
    argv = (process / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    if len(argv) != 5 or argv[1:4] != ['-m', 'deepseek_study.cli', 'run'] or argv[4] != str(study):
        raise ValueError('Launcher command does not match the intended study')
    environment = dict(
        value.split('=', 1)
        for value in (process / 'environ').read_bytes().decode().split('\0')
        if '=' in value
    )
    if environment.get('SLURM_JOB_ID') != str(job_id):
        raise ValueError('Launcher environment belongs to another Slurm job')
    groups = (process / 'cgroup').read_text()
    if re.search(rf'(^|/)job_{re.escape(str(job_id))}(/|$)', groups, re.MULTILINE) is not None:
        ownership = {'method': 'job_specific_cgroup'}
    else:
        ownership = slurm_ancestor(pid, job_id, proc_root)
    if (process / 'stat').read_text().rsplit(')', 1)[1].split()[19] != str(start_time):
        raise ValueError('Launcher start time changed during Slurm ownership validation')
    return {
        'pid': pid, 'start_time': str(start_time), 'job_id': str(job_id), 'argv': argv,
        'slurm_ownership': ownership,
    }


class IncompleteJournal(RuntimeError):
    pass


def latest_update(path):
    if not path.exists():
        return 0
    previous = 0
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                raise IncompleteJournal('Update journal currently has an incomplete final record')
            step = json.loads(line).get('step')
            if type(step) is not int or step <= previous:
                raise ValueError('Update journal is not strictly increasing')
            previous = step
    return previous


def validate_checkpoint(output, target, study, run):
    checkpoint = output / 'checkpoints' / f'step_{target}'
    marker_path = checkpoint / 'study/complete.json'
    if not marker_path.exists():
        return None
    marker = owned_json(marker_path)
    expected = {
        'format': 2,
        'step': target,
        'lag': study['lag'],
        'config_sha256': run['config_sha256'],
        'identity_sha256': run['identity_sha256'],
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise ValueError('Completed checkpoint does not match this study and selected step')
    manifest = checkpoint / 'study/components.json'
    if digest(manifest) != marker.get('components_sha256'):
        raise ValueError('Checkpoint component manifest checksum differs')
    records = json.loads(manifest.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError('Checkpoint component manifest is empty')
    paths = set()
    for record in records:
        name = record['path']
        path = checkpoint / name
        if name in paths or path.is_symlink() or not path.resolve().is_relative_to(checkpoint.resolve()):
            raise ValueError('Checkpoint component path is unsafe or duplicated')
        paths.add(name)
        if not path.is_file() or path.stat().st_size != record['size']:
            raise ValueError('Checkpoint component is missing or incomplete')
        if 'sha256' in record and digest(path) != record['sha256']:
            raise ValueError('Checkpoint component checksum differs')
    required = {'trainer/.metadata', 'orchestrator/progress.pt'}
    required.update(f'rng/rank_{rank}.pt' for rank in range(study['trainer_gpus']))
    if (
        not required.issubset(paths)
        or {name for name in paths if name.startswith('rng/')} != {name for name in required if name.startswith('rng/')}
        or not any(name.startswith('trainer/') and name != 'trainer/.metadata' for name in paths)
    ):
        raise ValueError('Checkpoint omits trainer, sampler or rank RNG state')
    queue = checkpoint / 'study/queue.pkl'
    if queue.is_symlink() or digest(queue) != marker.get('queue_sha256'):
        raise ValueError('Checkpoint queue checksum differs')
    if owned_json(marker_path) != marker:
        raise ValueError('Checkpoint marker changed while validating')
    return {'path': str(checkpoint), 'marker': marker, 'marker_sha256': digest(marker_path)}


def signal_launcher(pidfd, pid, start_time, study, job_id):
    process_record(pid, start_time, study, job_id)
    signal.pidfd_send_signal(pidfd, signal.SIGTERM)


def watch(args):
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Linux pidfd support is required to signal the exact verified launcher')
    if not 0 < args.timeout <= 86400 or not 0.02 <= args.poll_interval <= 1 or not 0 < args.exit_timeout <= 600:
        raise ValueError('Watch and exit deadlines must be bounded')
    if os.environ.get('SLURM_JOB_ID') != str(args.job_id):
        raise ValueError('Checkpoint watcher must run inside the original Slurm allocation')
    if args.receipt.exists() or args.intent.exists():
        raise FileExistsError('Checkpoint stop intent or receipt already exists')
    supervisor = owned_json(args.supervisor)
    study = owned_json(args.study)
    output = Path(study['output_dir'])
    if (
        str(supervisor.get('job_id')) != str(args.job_id)
        or supervisor.get('training_pid') != args.pid
        or supervisor.get('local_output') != str(output)
    ):
        raise ValueError('Supervisor does not identify the selected launcher and output')
    if args.target_step < study['lag'] or args.target_step > study['max_steps']:
        raise ValueError('Selected checkpoint must be post-bootstrap and within the training horizon')
    if args.target_step != study['max_steps'] and args.target_step % study['checkpoint_interval']:
        raise ValueError('Selected step is not a scheduled recovery checkpoint')
    run = owned_json(output / 'run.json')
    if (output / 'run-status.json').exists():
        raise RuntimeError('Launcher already has a shutdown receipt; refusing another stop')
    process = process_record(args.pid, args.start_time, args.study, args.job_id)
    pidfd = os.pidfd_open(args.pid)
    result = {
        'process': process, 'target_step': args.target_step, 'output': str(output), 'signal_sent': False,
        'study_sha256': digest(args.study), 'supervisor_sha256': digest(args.supervisor),
        'run_sha256': digest(output / 'run.json'),
    }
    try:
        process_record(args.pid, args.start_time, args.study, args.job_id)
        intent = {**result, 'status': 'waiting', 'created_at': time.time(), 'timeout_seconds': args.timeout}
        immutable_json(args.intent, intent)
        deadline = time.monotonic() + args.timeout
        while True:
            process_record(args.pid, args.start_time, args.study, args.job_id)
            try:
                step = latest_update(output / 'updates.jsonl')
            except IncompleteJournal:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Update journal did not settle before the deadline; launcher was not signalled')
                time.sleep(args.poll_interval)
                continue
            if step > args.target_step:
                raise RuntimeError('Selected checkpoint is behind committed updates; launcher was not signalled')
            if step == args.target_step:
                checkpoint = validate_checkpoint(output, args.target_step, study, run)
                if checkpoint is not None:
                    if latest_update(output / 'updates.jsonl') != args.target_step:
                        raise RuntimeError('Updates advanced during checkpoint validation; launcher was not signalled')
                    process_record(args.pid, args.start_time, args.study, args.job_id)
                    if latest_update(output / 'updates.jsonl') != args.target_step:
                        raise RuntimeError('Updates advanced before signalling; launcher was not signalled')
                    if any(
                        digest(path) != result[key]
                        for path, key in (
                            (args.study, 'study_sha256'), (args.supervisor, 'supervisor_sha256'),
                            (output / 'run.json', 'run_sha256'),
                        )
                    ):
                        raise ValueError('Study, supervisor or run identity changed before signalling')
                    result.update(checkpoint=checkpoint, signal_requested_at=time.time())
                    signal_launcher(pidfd, args.pid, args.start_time, args.study, args.job_id)
                    result.update(signal_sent=True, signal='SIGTERM')
                    poller = select.poll()
                    poller.register(pidfd, select.POLLIN)
                    if not poller.poll(round(args.exit_timeout * 1000)):
                        raise TimeoutError('Launcher did not exit within the deadline; no stronger signal was sent')
                    final_step = latest_update(output / 'updates.jsonl')
                    result['final_committed_step'] = final_step
                    if final_step != args.target_step:
                        raise RuntimeError('A later update committed during shutdown; do not claim lossless migration')
                    final_checkpoint = validate_checkpoint(output, args.target_step, study, run)
                    if final_checkpoint != checkpoint:
                        raise RuntimeError('Checkpoint disappeared or changed during launcher shutdown')
                    shutdown = owned_json(output / 'run-status.json')
                    if (
                        shutdown.get('status') != 'killed'
                        or shutdown.get('received_signum') != signal.SIGTERM
                        or str(shutdown.get('slurm_job_id')) != str(args.job_id)
                        or shutdown.get('cleanup_errors') != []
                    ):
                        raise RuntimeError('Launcher shutdown did not confirm clean owned-process cleanup')
                    result['launcher_shutdown'] = shutdown
                    result.update(status='launcher_stopped_at_checkpoint', finished_at=time.time())
                    immutable_json(args.receipt, result)
                    return result
            if time.monotonic() >= deadline:
                raise TimeoutError('Selected checkpoint was not ready before the deadline; launcher was not signalled')
            time.sleep(args.poll_interval)
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}', finished_at=time.time())
        if not args.receipt.exists():
            immutable_json(args.receipt, result)
        raise
    finally:
        os.close(pidfd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--supervisor', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--start-time', required=True)
    parser.add_argument('--job-id', type=int, required=True)
    parser.add_argument('--target-step', type=int, required=True)
    parser.add_argument('--intent', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=10800)
    parser.add_argument('--exit-timeout', type=float, default=180)
    parser.add_argument('--poll-interval', type=float, default=0.1)
    args = parser.parse_args()
    print(json.dumps(watch(args)), flush=True)


if __name__ == '__main__':
    main()
