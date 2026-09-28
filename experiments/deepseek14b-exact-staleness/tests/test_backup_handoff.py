import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
pytestmark = pytest.mark.skipif(not Path('/proc/self/stat').exists(), reason='Linux process handoff')


def wait_for(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError('Timed out waiting for backup handoff')


@pytest.mark.parametrize('finish', ['normal', 'kill_handoff'])
def test_live_handoff_preserves_original_supervision_and_restores_it(tmp_path, finish):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'run.json').write_text(json.dumps({'run_uuid': 'test', 'config_sha256': 'config', 'identity_sha256': 'identity'}))
    (source / 'updates.jsonl').write_text('{"step":1}\n')
    destination, stop, status = tmp_path / 'shared', tmp_path / 'stop', tmp_path / 'status.json'
    original = subprocess.Popen([sys.executable, str(SCRIPTS / 'local_backup.py'), '--source', str(source), '--destination', str(destination), '--stop-file', str(stop), '--status', str(status), '--lock', str(tmp_path / 'lock'), '--interval', '1'])
    worker = None
    try:
        wait_for(lambda: status.exists())
        supervisor = tmp_path / 'supervisor.json'
        supervisor.write_text(json.dumps({'backup_pid': original.pid, 'training_pid': os.getpid(), 'local_output': str(source), 'shared_backup': str(destination), 'job_id': 'test'}))
        receipt = tmp_path / 'handoff.json'
        worker = subprocess.Popen([sys.executable, str(SCRIPTS / 'backup_handoff.py'), '--supervisor', str(supervisor), '--script', str(SCRIPTS / 'local_backup.py'), '--receipt', str(receipt)], env={**os.environ, 'SLURM_JOB_ID': 'test'})
        wait_for(lambda: receipt.exists())
        assert json.loads(receipt.read_text())['status'] == 'active'
        assert (Path('/proc') / str(original.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'T'
        (source / 'updates.jsonl').write_text('{"step":1}\n{"step":2}\n')
        wait_for(lambda: (destination / 'updates.jsonl').read_text().endswith('{"step":2}\n'))
        if finish == 'kill_handoff':
            replacement_pid = json.loads(receipt.read_text())['replacement_pid']
            worker.kill()
            worker.wait(timeout=10)
            wait_for(lambda: (Path('/proc') / str(original.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0] != 'T')
            replacement_stat = Path('/proc') / str(replacement_pid) / 'stat'
            assert not replacement_stat.exists() or replacement_stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
        stop.touch()
        if finish == 'normal':
            assert worker.wait(timeout=20) == 0
            assert json.loads(receipt.read_text())['status'] == 'completed'
        assert original.wait(timeout=20) == 0
    finally:
        stop.touch()
        if worker is not None and worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=20)
        if original.poll() is None:
            os.kill(original.pid, signal.SIGCONT)
            original.wait(timeout=20)
