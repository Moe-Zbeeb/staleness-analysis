import json
import os
import signal
import subprocess
from types import SimpleNamespace

import pytest
import torch

from deepseek_study.runtime import launcher, processes


@pytest.fixture
def mocked_launch(study, monkeypatch):
    monkeypatch.setattr(launcher, "configure_nccl_transport", lambda: None)
    monkeypatch.setattr(launcher, "verify_upstream", lambda root: None)
    monkeypatch.setattr(launcher, "validate_prepared", lambda study: {})
    monkeypatch.setattr(launcher, "capture", lambda root, study: {"sha256": "source", "runtime": {"vllm": "test"}})
    monkeypatch.setattr(launcher, "snapshot", lambda root, dest, identity: (dest / "src").mkdir(parents=True))
    monkeypatch.setattr(launcher, "build", lambda *args: None)
    monkeypatch.setattr(launcher, "env_servers", lambda config: ())
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "mock")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(total_memory=1))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 0))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("DEEPSEEK_STUDY_RUNBOARD", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    return study


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("observer_kill_error", [ProcessLookupError, PermissionError])
def test_launcher_retains_failure_and_finishes_owned_cleanup_after_repeated_signal(
    mocked_launch, tmp_path, monkeypatch, capsys, interrupted, observer_kill_error
):
    study = mocked_launch
    started = []
    signalled = []
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}

    class Process:
        def __init__(self, args, **kwargs):
            assert kwargs["start_new_session"] is True
            assert "deepseek_study.tracking.runboard" not in args
            assert kwargs["env"]["DEEPSEEK_STUDY_RUNBOARD"] == "0"
            self.pid = 12340 + len(started)
            self.controller = "controller" in args
            self.observer = "deepseek_study.tracking.observer" in args
            self.waits = 0
            self.polls = 0
            started.append(self)

        def poll(self):
            self.polls += 1
            if self.controller and interrupted and self.polls == 1:
                os.kill(os.getpid(), signal.SIGTERM)
            return 1 if self.controller else None

        def wait(self, timeout=None):
            assert timeout is not None
            assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
            assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
            os.kill(os.getpid(), signal.SIGTERM)
            self.waits += 1
            if self.observer and self.waits == 1:
                raise subprocess.TimeoutExpired("observer", timeout)
            return 1 if self.controller else 0

    def killpg(pid, signum):
        assert pid in {process.pid for process in started}
        assert pid != os.getpgrp()
        signalled.append((pid, signum))
        if next(process for process in started if process.pid == pid).observer:
            raise observer_kill_error("observer cleanup race")

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    monkeypatch.setattr(processes.os, "killpg", killpg)
    expected_type = KeyboardInterrupt if interrupted else RuntimeError
    expected_message = "Launcher received SIGTERM" if interrupted else "controller exited with status 1"
    with pytest.raises(expected_type, match=expected_message):
        launcher.launch(study, tmp_path)
    status = json.loads((study.output_dir / "run-status.json").read_text())
    assert status["status"] == ("killed" if interrupted else "crashed")
    assert status["error"]["type"] == expected_type.__name__
    assert expected_message in status["error"]["message"]
    assert status["received_signal"] == ("SIGTERM" if interrupted else None)
    assert status["received_signum"] == (signal.SIGTERM if interrupted else None)
    assert status["slurm_job_id"] == "1234"
    assert status["process_exit_codes_before_cleanup"]["controller"] == 1
    assert status["maximum_updates"] == study.max_steps
    assert status["checkpoint_interval"] == study.checkpoint_interval
    assert status["cleanup_errors"] == []
    assert json.loads((study.output_dir / "paper-status.json").read_text())["status"] == "interrupted"
    assert {signum: signal.getsignal(signum) for signum in previous} == previous
    assert all(process.waits for process in started)
    assert signalled
    if observer_kill_error is PermissionError:
        assert "Cleanup warning: paper-metrics: SIGKILL failed: PermissionError" in capsys.readouterr().err


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_during_spawn_is_deferred_until_child_is_registered(mocked_launch, tmp_path, monkeypatch, signum):
    study = mocked_launch
    started = []
    signalled = []
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])

    class Process:
        def __init__(self, args, **kwargs):
            self.pid = 12340 + len(started)
            self.waits = 0
            started.append(self)
            assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == previous_mask
            if len(started) == 3:
                os.kill(os.getpid(), signum)
                os.kill(os.getpid(), signal.SIGINT if signum == signal.SIGTERM else signal.SIGTERM)

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.waits += 1
            return 0

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    monkeypatch.setattr(processes.os, "killpg", lambda pid, sent: signalled.append((pid, sent)))
    with pytest.raises(KeyboardInterrupt, match=signal.Signals(signum).name):
        launcher.launch(study, tmp_path)
    status = json.loads((study.output_dir / "run-status.json").read_text())
    assert status["status"] == "killed"
    assert status["received_signum"] == signum
    assert status["received_signal"] == signal.Signals(signum).name
    assert set(status["process_exit_codes_before_cleanup"]) == {"paper-metrics", "tensorboard", "inference"}
    assert len(started) == 3 and all(process.waits for process in started)
    assert signalled == [(started[-1].pid, signal.SIGTERM), (started[-1].pid, signal.SIGKILL)]
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == previous_mask


def test_tensorboard_drains_after_terminal_state_and_final_paper_metrics(mocked_launch, tmp_path, monkeypatch):
    study = mocked_launch
    started = []
    signalled = []
    drained = []

    class Process:
        def __init__(self, args, **kwargs):
            assert "deepseek_study.tracking.runboard" not in args
            self.pid = 12340 + len(started)
            self.name = (
                "paper" if "deepseek_study.tracking.observer" in args else
                "tensorboard" if "deepseek_study.tracking.tensorboard" in args else
                "controller" if "controller" in args else
                "trainer" if "deepseek_study.runtime.trainer" in args else "inference"
            )
            if self.name == "controller":
                (study.output_dir / "study-complete.json").write_text(json.dumps({"step": study.max_steps}))
            started.append(self)

        def poll(self):
            return 0 if self.name in {"controller", "trainer"} else None

        def wait(self, timeout=None):
            assert timeout is not None
            if self.name == "paper":
                assert json.loads((study.output_dir / "run-status.json").read_text())["status"] == "finished"
                (study.output_dir / "paper-metrics.jsonl").write_text('{"step":70,"metrics":{"reward":1}}\n')
                (study.output_dir / "paper-status.json").write_text('{"status":"complete"}')
                drained.append("paper")
            elif self.name == "tensorboard":
                assert drained == ["paper"]
                assert json.loads((study.output_dir / "paper-status.json").read_text())["status"] == "complete"
                assert json.loads((study.output_dir / "paper-metrics.jsonl").read_text())["step"] == 70
                drained.append("tensorboard")
            return 0

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    monkeypatch.setattr(processes.os, "killpg", lambda pid, signum: signalled.append(pid))
    launcher.launch(study, tmp_path)
    assert drained == ["paper", "tensorboard"]
    assert all(process.pid not in signalled for process in started if process.name in {"paper", "tensorboard"})


def test_owned_cleanup_is_bounded_even_when_a_process_does_not_exit(monkeypatch):
    timeouts = []
    signals = []

    class Process:
        pid = 12345

        def wait(self, timeout=None):
            timeouts.append(timeout)
            raise subprocess.TimeoutExpired("worker", timeout)

    monkeypatch.setattr(processes.os, "killpg", lambda pid, signum: signals.append((pid, signum)))
    errors = processes.stop_process_groups({"worker": Process()}, graceful_timeout=0.1, kill_timeout=0.1)
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
    assert len(timeouts) == 2 and all(0 <= value <= 0.1 for value in timeouts)
    assert errors == ["worker: still running after the shutdown deadline"]


def test_process_group_signal_errors_are_reported_without_masking_failure(monkeypatch):
    def deny(*args):
        raise PermissionError("cannot signal owned process")

    monkeypatch.setattr(processes.os, "killpg", deny)
    process = SimpleNamespace(pid=12345, wait=lambda timeout: 0)
    errors = processes.stop_process_groups({"worker": process})
    assert len(errors) == 2
    assert all("PermissionError" in error for error in errors)
