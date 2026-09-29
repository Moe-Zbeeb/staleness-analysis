import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/stop_at_checkpoint.py'
SPEC = importlib.util.spec_from_file_location('stop_at_checkpoint', SCRIPT)
watcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watcher)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def checkpoint(output, study, run, step=425):
    root = output / 'checkpoints' / f'step_{step}'
    records = []
    for name in ['trainer/.metadata', 'trainer/shard.distcp', 'orchestrator/progress.pt', *[f'rng/rank_{i}.pt' for i in range(4)]]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'checkpoint-bytes')
        records.append({'path': name, 'size': path.stat().st_size, 'sha256': watcher.digest(path)})
    manifest = root / 'study/components.json'
    write(manifest, records)
    queue = root / 'study/queue.pkl'
    queue.write_bytes(b'not-a-valid-pickle-is-never-loaded')
    marker = {
        'format': 2, 'step': step, 'lag': study['lag'], 'config_sha256': run['config_sha256'],
        'identity_sha256': run['identity_sha256'], 'components_sha256': watcher.digest(manifest),
        'queue_sha256': watcher.digest(queue),
    }
    write(root / 'study/complete.json', marker)
    return root


@pytest.fixture
def setup(tmp_path, monkeypatch):
    output = tmp_path / 'output'
    output.mkdir()
    study = {'output_dir': str(output), 'lag': 256, 'max_steps': 1000, 'checkpoint_interval': 25, 'trainer_gpus': 4}
    run = {'config_sha256': 'config', 'identity_sha256': 'identity'}
    args = SimpleNamespace(
        supervisor=tmp_path / 'supervisor.json', study=tmp_path / 'study.json', pid=71, start_time='100',
        job_id=2145464, target_step=425, intent=tmp_path / 'intent.json', receipt=tmp_path / 'receipt.json',
        timeout=1, exit_timeout=1, poll_interval=0.02,
    )
    write(args.study, study)
    write(output / 'run.json', run)
    write(args.supervisor, {'job_id': str(args.job_id), 'training_pid': args.pid, 'local_output': str(output)})
    (output / 'updates.jsonl').write_text('{"step":424}\n{"step":425}\n')
    root = checkpoint(output, study, run)
    signals = []
    monkeypatch.setenv('SLURM_JOB_ID', str(args.job_id))
    monkeypatch.setattr(watcher, 'process_record', lambda *a: {'pid': args.pid, 'start_time': '100'})
    monkeypatch.setattr(watcher.os, 'pidfd_open', lambda pid: os.open(os.devnull, os.O_RDONLY), raising=False)
    def receive_signal(*a):
        signals.append(a)
        write(output / 'run-status.json', {
            'status': 'killed', 'received_signum': watcher.signal.SIGTERM,
            'slurm_job_id': str(args.job_id), 'cleanup_errors': [],
        })
    monkeypatch.setattr(watcher.signal, 'pidfd_send_signal', receive_signal, raising=False)
    monkeypatch.setattr(watcher.select, 'poll', lambda: SimpleNamespace(register=lambda *a: None, poll=lambda *a: [1]), raising=False)
    return args, output, study, run, root, signals


def test_stop_signals_only_exact_launcher_after_complete_checkpoint(setup):
    args, output, study, run, root, signals = setup
    result = watcher.watch(args)
    assert result['status'] == 'launcher_stopped_at_checkpoint'
    assert result['final_committed_step'] == 425
    assert result['signal_sent'] is True
    assert len(signals) == 1 and signals[0][1] == watcher.signal.SIGTERM
    assert json.loads(args.intent.read_text())['status'] == 'waiting'
    assert json.loads(args.receipt.read_text()) == result
    assert watcher.digest(root / 'study/queue.pkl') == result['checkpoint']['marker']['queue_sha256']


@pytest.mark.parametrize('failure', ['passed_step', 'wrong_checkpoint', 'corrupt_queue', 'missing_rng', 'wrong_supervisor', 'wrong_job'])
def test_refuses_unsafe_boundaries_without_signalling(setup, monkeypatch, failure):
    args, output, study, run, root, signals = setup
    if failure == 'passed_step':
        (output / 'updates.jsonl').write_text('{"step":425}\n{"step":426}\n')
    elif failure == 'wrong_checkpoint':
        marker = json.loads((root / 'study/complete.json').read_text())
        marker['identity_sha256'] = 'another-run'
        write(root / 'study/complete.json', marker)
    elif failure == 'corrupt_queue':
        (root / 'study/queue.pkl').write_bytes(b'corrupt')
    elif failure == 'missing_rng':
        (root / 'rng/rank_3.pt').unlink()
    elif failure == 'wrong_supervisor':
        write(args.supervisor, {'job_id': str(args.job_id), 'training_pid': 999, 'local_output': str(output)})
    else:
        monkeypatch.setenv('SLURM_JOB_ID', 'another')
    with pytest.raises((ValueError, RuntimeError)):
        watcher.watch(args)
    assert signals == []


def test_recheck_prevents_kill_if_update_commits_during_validation(setup, monkeypatch):
    args, output, study, run, root, signals = setup
    validate = watcher.validate_checkpoint
    def advancing(*a):
        value = validate(*a)
        (output / 'updates.jsonl').write_text('{"step":425}\n{"step":426}\n')
        return value
    monkeypatch.setattr(watcher, 'validate_checkpoint', advancing)
    with pytest.raises(RuntimeError, match='advanced'):
        watcher.watch(args)
    assert signals == []


def test_reports_commit_during_shutdown_instead_of_claiming_lossless_stop(setup, monkeypatch):
    args, output, study, run, root, signals = setup
    def signal_sent(*a):
        signals.append(a)
        (output / 'updates.jsonl').write_text('{"step":425}\n{"step":426}\n')
    monkeypatch.setattr(watcher.signal, 'pidfd_send_signal', signal_sent)
    with pytest.raises(RuntimeError, match='later update committed'):
        watcher.watch(args)
    receipt = json.loads(args.receipt.read_text())
    assert receipt['status'] == 'failed' and receipt['signal_sent']
    assert receipt['final_committed_step'] == 426
    assert len(signals) == 1


def test_pending_checkpoint_times_out_without_signal(setup):
    args, output, study, run, root, signals = setup
    (root / 'study/complete.json').unlink()
    args.timeout = 0.03
    with pytest.raises(TimeoutError):
        watcher.watch(args)
    assert signals == []


def test_receipts_are_never_overwritten(setup):
    args, output, study, run, root, signals = setup
    watcher.watch(args)
    original = args.receipt.read_bytes()
    with pytest.raises(FileExistsError):
        watcher.watch(args)
    assert args.receipt.read_bytes() == original and len(signals) == 1


@pytest.mark.parametrize('field', ['start', 'argv', 'environment', 'cgroup', 'uid', 'state'])
def test_process_identity_rejects_wrong_target(tmp_path, field):
    root = tmp_path / 'proc'
    process = root / '71'
    process.mkdir(parents=True)
    fields = ['S', *['0'] * 18, '100', '0']
    if field == 'start':
        fields[19] = '101'
    if field == 'state':
        fields[0] = 'Z'
    (process / 'stat').write_text('71 (python) ' + ' '.join(fields))
    uid = os.geteuid() + 1 if field == 'uid' else os.geteuid()
    (process / 'status').write_text('Uid:\t' + '\t'.join([str(uid)] * 4) + '\n')
    argv = ['python', '-m', 'deepseek_study.cli', 'run', '/study.json']
    if field == 'argv':
        argv[-1] = '/other.json'
    (process / 'cmdline').write_bytes(('\0'.join(argv) + '\0').encode())
    job = '999' if field == 'environment' else '2145464'
    (process / 'environ').write_bytes(f'SLURM_JOB_ID={job}\0'.encode())
    group = '21454640' if field == 'cgroup' else '2145464'
    (process / 'cgroup').write_text(f'0::/slurm/uid_{uid}/job_{group}/step_batch\n')
    with pytest.raises(ValueError):
        watcher.process_record(71, '100', Path('/study.json'), 2145464, root)


def test_valid_process_identity_returns_only_nonsensitive_target_metadata(tmp_path):
    root = tmp_path / 'proc'
    process = root / '71'
    process.mkdir(parents=True)
    (process / 'stat').write_text('71 (python) ' + ' '.join(['S', *['0'] * 18, '100', '0']))
    (process / 'status').write_text('Uid:\t' + '\t'.join([str(os.geteuid())] * 4) + '\n')
    (process / 'cmdline').write_bytes(b'python\0-m\0deepseek_study.cli\0run\0/study.json\0')
    (process / 'environ').write_bytes(b'SLURM_JOB_ID=2145464\0SECRET=not-returned\0')
    (process / 'cgroup').write_text('0::/slurm/job_2145464/step_batch\n')
    assert watcher.process_record(71, '100', Path('/study.json'), 2145464, root) == {
        'pid': 71, 'start_time': '100', 'job_id': '2145464',
        'argv': ['python', '-m', 'deepseek_study.cli', 'run', '/study.json'],
        'slurm_ownership': {'method': 'job_specific_cgroup'},
    }


def test_journal_rejects_partial_and_reordered_records(tmp_path):
    path = tmp_path / 'updates.jsonl'
    path.write_text('{"step":425}\n{"step":426')
    with pytest.raises(watcher.IncompleteJournal):
        watcher.latest_update(path)
    path.write_text('{"step":426}\n{"step":425}\n')
    with pytest.raises(ValueError):
        watcher.latest_update(path)


@pytest.mark.skipif(not hasattr(os, 'pidfd_open'), reason='Linux pidfd required')
def test_real_pidfd_stops_owned_child_without_signalling_another_process(tmp_path, monkeypatch):
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    descriptor = None
    try:
        descriptor = os.pidfd_open(child.pid)
        metadata = (Path('/proc') / str(child.pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        start = metadata[19]
        def validate(pid, start_time, study, job_id):
            path = Path('/proc') / str(pid)
            assert path.stat().st_uid == os.geteuid()
            assert path.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19] == start_time
            assert pid == child.pid and start_time == start
        monkeypatch.setattr(watcher, 'process_record', validate)
        watcher.signal_launcher(descriptor, child.pid, start, tmp_path / 'study.json', 2145464)
        assert child.wait(timeout=5) == -watcher.signal.SIGTERM
        assert unrelated.poll() is None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for process in (child, unrelated):
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


def test_checkpoint_disappearing_during_shutdown_is_a_failed_stop(setup, monkeypatch):
    args, output, study, run, root, signals = setup
    send = watcher.signal.pidfd_send_signal
    def disappear(*a):
        send(*a)
        (root / 'study/complete.json').unlink()
    monkeypatch.setattr(watcher.signal, 'pidfd_send_signal', disappear)
    with pytest.raises(RuntimeError, match='disappeared or changed'):
        watcher.watch(args)
    receipt = json.loads(args.receipt.read_text())
    assert receipt['signal_sent'] and receipt['status'] == 'failed'
    assert len(signals) == 1


@pytest.fixture
def ancestor_processes(tmp_path, monkeypatch):
    root = tmp_path / 'proc'
    records = {
        71: {'pid': 71, 'ppid': 70, 'uid': os.geteuid(), 'start_time': '100'},
        70: {'pid': 70, 'ppid': 60, 'uid': os.geteuid(), 'start_time': '90'},
        60: {'pid': 60, 'ppid': 1, 'uid': 0, 'start_time': '80'},
    }
    process = root / '60'
    process.mkdir(parents=True)
    (process / 'comm').write_text('slurmstepd\n')
    (process / 'cmdline').write_bytes(b'slurmstepd: [2145464.batch]\0')
    monkeypatch.setattr(watcher, 'ancestry_record', lambda pid, proc: dict(records[pid]))
    return root, records


def test_root_owned_exact_job_slurm_ancestor_is_valid(ancestor_processes):
    root, records = ancestor_processes
    result = watcher.slurm_ancestor(71, 2145464, root)
    assert result == {
        'method': 'root_owned_slurmstepd_ancestor', 'ancestor': records[60],
        'command': 'slurmstepd: [2145464.batch]',
    }


@pytest.mark.parametrize('failure', ['wrong_job', 'wrong_name', 'unowned_ancestor', 'other_user', 'cycle', 'reused_pid'])
def test_ancestor_fallback_does_not_trust_environment_only(ancestor_processes, monkeypatch, failure):
    root, records = ancestor_processes
    if failure == 'wrong_job':
        (root / '60/cmdline').write_bytes(b'slurmstepd: [21454640.batch]\0')
    elif failure == 'wrong_name':
        (root / '60/comm').write_text('bash\n')
    elif failure == 'unowned_ancestor':
        records[60]['uid'] = os.geteuid()
    elif failure == 'other_user':
        records[70]['uid'] = os.geteuid() + 1
    elif failure == 'cycle':
        records[70]['ppid'] = 71
    else:
        calls = []
        def reused(pid, proc):
            calls.append(pid)
            result = dict(records[pid])
            if pid == 71 and calls.count(71) > 1:
                result['start_time'] = '999'
            return result
        monkeypatch.setattr(watcher, 'ancestry_record', reused)
    with pytest.raises(ValueError):
        watcher.slurm_ancestor(71, 2145464, root)


def test_generic_slurmd_cgroup_uses_verified_root_ancestor(tmp_path, monkeypatch):
    root = tmp_path / 'proc'
    process = root / '71'
    process.mkdir(parents=True)
    (process / 'stat').write_text('71 (python) ' + ' '.join(['S', *['0'] * 18, '100', '0']))
    (process / 'status').write_text('Uid:\t' + '\t'.join([str(os.geteuid())] * 4) + '\n')
    (process / 'cmdline').write_bytes(b'python\0-m\0deepseek_study.cli\0run\0/study.json\0')
    (process / 'environ').write_bytes(b'SLURM_JOB_ID=2145464\0')
    (process / 'cgroup').write_text('0::/system.slice/slurmd.service\n')
    proof = {'method': 'root_owned_slurmstepd_ancestor', 'ancestor': {'pid': 60, 'uid': 0}}
    calls = []
    def ancestor(*args):
        calls.append(args)
        return proof
    monkeypatch.setattr(watcher, 'slurm_ancestor', ancestor)
    result = watcher.process_record(71, '100', Path('/study.json'), 2145464, root)
    assert result['slurm_ownership'] == proof
    assert calls == [(71, 2145464, root)]
