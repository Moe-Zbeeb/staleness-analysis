import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("warmup_launch", SCRIPTS / "launch_full_run.py")
LAUNCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCH)


def test_import_warmup_uses_same_interpreter_without_changing_parent_gpu_visibility(tmp_path, monkeypatch):
    environment = {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "OMP_NUM_THREADS": "4", "SLURM_JOB_ID": "123"}
    original = dict(environment)

    def run(command, **kwargs):
        assert command[:2] == [sys.executable, "-c"]
        assert kwargs["timeout"] == 180
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert kwargs["env"]["DEEPSEEK_STUDY_TORCH_IMPORT_TRACE_AFTER"] == "30"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {"torch_version": "pinned", "torch_file": "/runtime/torch/__init__.py", "cuda_visible_devices": ""}
            ),
            stderr='[torch-import-stage]{"stage":"before_torch_import","seconds":0}\n'
            '[torch-import-stage]{"stage":"after_torch_import","seconds":12}\n',
        )

    monkeypatch.setattr(LAUNCH.subprocess, "run", run)
    path = tmp_path / "warmup.json"
    result = LAUNCH.warm_torch_import(path, environment)
    assert result["status"] == "complete" and result["last_stage"] == "after_torch_import"
    assert json.loads(path.read_text()) == result
    assert environment == original
    with pytest.raises(FileExistsError, match="fresh launch attempt"):
        LAUNCH.warm_torch_import(path, environment)


@pytest.mark.parametrize("failure", ["timeout", "exit", "missing_acknowledgement", "visible_gpu", "spawn"])
def test_import_warmup_fails_closed_and_preserves_diagnostics(tmp_path, monkeypatch, failure):
    stage = '[torch-import-stage]{"stage":"before_torch_import","seconds":0}\n'

    def run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                command, 180, output=b"partial", stderr=(stage + "import traceback").encode()
            )
        if failure == "spawn":
            raise OSError("interpreter unavailable")
        if failure == "exit":
            return subprocess.CompletedProcess(command, 1, stdout="", stderr=stage + "ImportError: missing runtime")
        acknowledgement = (
            ""
            if failure == "missing_acknowledgement"
            else json.dumps(
                {
                    "torch_version": "pinned",
                    "torch_file": "/runtime/torch/__init__.py",
                    "cuda_visible_devices": "0",
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, stdout=acknowledgement, stderr=stage)

    monkeypatch.setattr(LAUNCH.subprocess, "run", run)
    path = tmp_path / "warmup.json"
    with pytest.raises(RuntimeError, match="warmup failed"):
        LAUNCH.warm_torch_import(path)
    result = json.loads(path.read_text())
    assert result["status"] == ("timeout" if failure == "timeout" else "failed")
    assert result["timeout_seconds"] == 180
    assert result["trace_after_seconds"] == 30
    assert result["error"]
    assert result["last_stage"] == (None if failure == "spawn" else "before_torch_import")
    if failure == "timeout":
        assert result["stdout_tail"] == "partial"
        assert "import traceback" in result["stderr_tail"]


def test_actual_cpu_only_torch_import_records_loaded_runtime(tmp_path):
    result = LAUNCH.warm_torch_import(tmp_path / "warmup.json")
    assert result["status"] == "complete"
    assert result["result"]["cuda_visible_devices"] == ""
    assert Path(result["result"]["torch_file"]).is_file()
    assert [stage["stage"] for stage in result["stages"]] == ["before_torch_import", "after_torch_import"]
