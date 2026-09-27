import os
import signal
import subprocess
import time


def signal_process_group(name, process, signum, errors):
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass
    except OSError as error:
        errors.append(f"{name}: {signal.Signals(signum).name} failed: {type(error).__name__}: {error}")


def wait_processes(processes, timeout, errors):
    deadline = time.monotonic() + timeout
    for name, process in processes.items():
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            errors.append(f"{name}: still running after the shutdown deadline")
        except OSError as error:
            errors.append(f"{name}: wait failed: {type(error).__name__}: {error}")


def stop_process_groups(processes, graceful_timeout=20, kill_timeout=5):
    errors = []
    for name, process in processes.items():
        signal_process_group(name, process, signal.SIGTERM, errors)
    graceful_errors = []
    wait_processes(processes, graceful_timeout, graceful_errors)
    for name, process in processes.items():
        signal_process_group(name, process, signal.SIGKILL, errors)
    wait_processes(processes, kill_timeout, errors)
    return errors


def drain_process(name, process, timeout=20, kill_timeout=5):
    errors = []
    try:
        process.wait(timeout=timeout)
        return True, errors
    except subprocess.TimeoutExpired:
        signal_process_group(name, process, signal.SIGKILL, errors)
        wait_processes({name: process}, kill_timeout, errors)
        return False, errors
    except OSError as error:
        errors.append(f"{name}: wait failed: {type(error).__name__}: {error}")
        return False, errors
