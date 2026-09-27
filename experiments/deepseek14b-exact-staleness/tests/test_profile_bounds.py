import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.parametrize("qwen", [False, True])
def test_profile_source_changes_only_model_pin(tmp_path, qwen):
    spec = importlib.util.spec_from_file_location("small_profile", SCRIPTS / "prepare_small_model_profile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    (source / "src/deepseek_study").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "vendor/prime-rl").mkdir(parents=True)
    init = source / "src/deepseek_study/__init__.py"
    init.write_text(f'MODEL_ID = "{module.BASE_MODEL_ID}"\nMODEL_REVISION = "{module.BASE_MODEL_REVISION}"\n')
    other = source / "src/deepseek_study/loss.py"
    other.write_text("value = 1\n")
    assets = source / "src/deepseek_study/dataset/assets.py"
    assets.parent.mkdir()
    assets.write_bytes((SCRIPTS.parent / "src/deepseek_study/dataset/assets.py").read_bytes())
    (source / "pyproject.toml").write_text("")
    hashes = {str(path.relative_to(source)): module.digest(path) for path in (init, other, assets)}
    (source / "PACKAGE_SHA256.json").write_text(json.dumps(hashes))
    destination = tmp_path / "profile"
    model_id = module.QWEN_MODEL_ID if qwen else module.MODEL_ID
    revision = module.QWEN_MODEL_REVISION if qwen else module.MODEL_REVISION
    expected = ["src/deepseek_study/__init__.py"]
    if qwen:
        expected.append("src/deepseek_study/dataset/assets.py")
    assert module.clone_source(source, destination, model_id, revision) == expected
    assert (destination / "src/deepseek_study/loss.py").read_bytes() == other.read_bytes()
    assert model_id in (destination / "src/deepseek_study/__init__.py").read_text()
    generated = (destination / "src/deepseek_study/dataset/assets.py").read_text()
    assert ('endswith("<think>\\n")' in generated) == (not qwen)
    assert ("original.bos_token_id, original.eos_token_id" in generated) == qwen
    assert module.BASE_MODEL_ID in init.read_text()
    other.write_text("value = 2\n")
    with pytest.raises(ValueError, match="Frozen baseline source changed"):
        module.clone_source(source, tmp_path / "bad")


@pytest.mark.parametrize(
    "mode,expected", [("success", "completed"), ("failure", "incomplete"), ("timeout", "incomplete")]
)
def test_bounded_profile_terminates_and_distinguishes_failure(tmp_path, mode, expected):
    package = tmp_path / "release/src/deepseek_study"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "cli.py").write_text(
        "import json, pathlib, sys, time\n"
        "study = json.loads(pathlib.Path(sys.argv[2]).read_text())\n"
        "output = pathlib.Path(study['output_dir'])\n"
        "output.mkdir()\n"
        f"mode = {mode!r}\n"
        "if mode == 'failure':\n"
        "    raise SystemExit(7)\n"
        "if mode == 'success':\n"
        "    (output / 'updates.jsonl').write_text(json.dumps({'step': 4}) + '\\n')\n"
        "    time.sleep(0.8)\n"
        "    (output / 'metrics.jsonl').write_text(json.dumps({'producer': 'trainer', 'step': 4, 'time/forward_backward': 1.0}) + '\\n')\n"
        "while True:\n"
        "    time.sleep(0.1)\n"
    )
    study = {
        "lag": 256,
        "max_steps": 1000,
        "checkpoint_interval": 100,
        "output_dir": str(tmp_path / "run"),
        "metrics_mirror_root": str(tmp_path / "mirror"),
    }
    (tmp_path / "study.json").write_text(json.dumps(study))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_bounded_profile.py"),
            "--directory",
            str(tmp_path),
            "--updates",
            "4",
            "--seconds",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    report = json.loads((tmp_path / "timing-summary.json").read_text())
    assert report["status"] == expected, result.stderr
    assert (result.returncode == 0) == (mode == "success")
    assert report["completed_updates"] == (4 if mode == "success" else 0)
    assert report["measures_exact256_phase"] is False
    assert json.loads((tmp_path / "study.json").read_text())["max_steps"] == 1000
