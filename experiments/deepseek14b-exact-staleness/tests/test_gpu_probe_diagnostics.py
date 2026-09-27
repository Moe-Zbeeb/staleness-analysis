import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_allocated_gpus.py"
SPEC = importlib.util.spec_from_file_location("gpu_probe_diagnostics", SCRIPT)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


@pytest.mark.parametrize("as_bytes", [False, True])
def test_timeout_preserves_last_stage_and_partial_traceback(monkeypatch, as_bytes):
    stdout = "partial stdout"
    stderr = '[gpu-probe-stage]{"stage":"before_torch_import","seconds":0.01}\nThread: blocked in import\n'
    if as_bytes:
        stdout, stderr = stdout.encode(), stderr.encode()

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 600
        assert kwargs["env"]["DEEPSEEK_STUDY_GPU_PROBE_TRACE_AFTER"] == "60"
        raise subprocess.TimeoutExpired(args[0], 600, output=stdout, stderr=stderr)

    monkeypatch.setattr(PROBE.subprocess, "run", timeout)
    result = PROBE.probe_device("gpu-1", timeout=600, trace_after_seconds=60)
    assert result["error"] == "probe_timeout"
    assert result["healthy"] is False
    assert result["last_stage"] == "before_torch_import"
    assert result["stdout_tail"] == "partial stdout"
    assert "blocked in import" in result["stderr_tail"]
    assert result["stages"][0]["seconds"] == 0.01


def test_probe_defaults_and_successful_result_are_preserved(monkeypatch):
    metadata = {"name": "NVIDIA A100", "bytes": 85_100_000_000, "uuid": "gpu-1", "bf16_backward": True}

    def complete(*args, **kwargs):
        assert kwargs["timeout"] == 120
        assert kwargs["env"]["DEEPSEEK_STUDY_GPU_PROBE_TRACE_AFTER"] == "0"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "7"
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(metadata) + "\n",
            stderr='[gpu-probe-stage]{"stage":"complete","seconds":1.2}\n',
        )

    monkeypatch.setattr(PROBE.subprocess, "run", complete)
    result = PROBE.probe_device("7")
    assert result["healthy"] is True and result["device"] == "7"
    assert all(result[key] == value for key, value in metadata.items())
    assert result["last_stage"] == "complete"


def test_occupied_gpu_error_remains_visible(monkeypatch):
    error = "RuntimeError: GPU is already occupied: 10% free"
    monkeypatch.setattr(
        PROBE.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr=error),
    )
    result = PROBE.probe_device("1")
    assert not result["healthy"]
    assert "GPU is already occupied:" in result["error"]


def test_probe_diagnostics_are_bounded_and_accept_incomplete_stage_lines():
    errors = '[gpu-probe-stage]{"stage":"before_mem_info","seconds":2}\n' + "trace" * 3000
    errors += '\n[gpu-probe-stage]{"stage":'
    result = PROBE.probe_diagnostics(b"x" * 10000, errors.encode())
    assert len(result["stdout_tail"]) == 4000
    assert len(result["stderr_tail"]) == 8000
    assert result["last_stage"] == "before_mem_info"
    assert PROBE.probe_diagnostics(None, None)["last_stage"] is None


def test_probe_announces_import_before_loading_torch():
    assert PROBE.PROBE.index("stage('before_torch_import')") < PROBE.PROBE.index("import torch")
    assert "free_bytes < 0.9 * total_bytes" in PROBE.PROBE
    compile(PROBE.PROBE, "gpu_probe_child", "exec")
