import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from local_backup import atomic_json, digest


def process_identity(pid):
    path = Path('/proc') / str(pid)
    if path.stat().st_uid != os.getuid():
        raise ValueError('Backup process belongs to another user')
    return (path / 'stat').read_text().rsplit(')', 1)[1].split()[19]


def resume(pid, identity):
    try:
        if process_identity(pid) == identity:
            os.kill(pid, signal.SIGCONT)
    except ProcessLookupError:
        pass
    except FileNotFoundError:
        pass


def handoff(supervisor, script, receipt):
    metadata = json.loads(supervisor.read_text())
    pid = metadata['backup_pid']
    identity = process_identity(pid)
    argv = (Path('/proc') / str(pid) / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    if len(argv) < 3 or Path(argv[1]).name != 'local_backup.py':
        raise ValueError('Expected the supervised local backup process')
    def argument(name):
        return argv[argv.index(name) + 1]
    if argument('--source') != metadata['local_output'] or argument('--destination') != metadata['shared_backup']:
        raise ValueError('Backup process arguments differ from supervisor')
    if os.environ.get('SLURM_JOB_ID') != metadata['job_id']:
        raise ValueError('Backup handoff must run within its training allocation')
    stop = Path(argument('--stop-file'))
    if stop.exists():
        raise ValueError('Training shutdown has already started')
    replacement = [argv[0], str(script), *argv[2:]]
    reader, writer = os.pipe()
    guardian = os.fork()
    if guardian == 0:
        os.close(writer)
        payload = b''
        try:
            while block := os.read(reader, 4096):
                payload += block
            if payload:
                child_record = json.loads(payload)
                child_pid = child_record['pid']
                try:
                    if process_identity(child_pid) == child_record['identity']:
                        os.kill(child_pid, signal.SIGINT)
                        deadline = time.monotonic() + 15
                        while time.monotonic() < deadline:
                            state = (Path('/proc') / str(child_pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0]
                            if state == 'Z':
                                break
                            time.sleep(0.1)
                        else:
                            if process_identity(child_pid) == child_record['identity']:
                                os.kill(child_pid, signal.SIGKILL)
                except (FileNotFoundError, ProcessLookupError):
                    pass
        finally:
            resume(pid, identity)
            os._exit(0)
    os.close(reader)
    child = None
    interrupted = False
    def interrupt(*_):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    record = {'original_backup_pid': pid, 'original_start_time': identity, 'training_pid': metadata['training_pid'], 'job_id': metadata['job_id'], 'replacement_script': str(script), 'replacement_sha256': digest(script)}
    try:
        with Path(argument('--lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if process_identity(pid) != identity:
                raise ValueError('Backup process identity changed')
            os.kill(pid, signal.SIGSTOP)
            deadline = time.monotonic() + 5
            while (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0] not in {'T', 't'}:
                if time.monotonic() > deadline:
                    raise RuntimeError('Original backup process did not pause')
                time.sleep(0.05)
            child = subprocess.Popen(replacement, start_new_session=True)
            os.write(writer, json.dumps({'pid': child.pid, 'identity': process_identity(child.pid)}).encode())
            record.update(status='active', replacement_pid=child.pid, started_at=time.time())
            atomic_json(receipt, record)
        while child.poll() is None:
            if interrupted:
                raise RuntimeError('Backup handoff was interrupted')
            if process_identity(pid) != identity:
                raise RuntimeError('Original backup process disappeared')
            time.sleep(1)
        if child.returncode or not stop.exists():
            raise RuntimeError(f'Replacement backup exited unexpectedly: {child.returncode}')
        record.update(status='completed', finished_at=time.time())
    except BaseException as error:
        record.update(status='fallback_to_original', error=repr(error), finished_at=time.time())
        raise
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        resume(pid, identity)
        os.close(writer)
        os.waitpid(guardian, 0)
        atomic_json(receipt, record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--supervisor', type=Path, required=True)
    parser.add_argument('--script', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError('A backup handoff receipt already exists')
    handoff(args.supervisor, args.script, args.receipt)


if __name__ == '__main__':
    main()
