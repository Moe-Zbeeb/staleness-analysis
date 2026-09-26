import json
import subprocess
import sys


def test_cli_creates_one_requested_run_and_builds_real_official_configs(tmp_path):
    config = tmp_path / "run.json"
    command = [sys.executable, "-m", "deepseek_study.cli"]
    subprocess.run(
        command + ["init", str(config), "--lag", "7", "--root", str(tmp_path / "project")],
        check=True,
        capture_output=True,
    )
    value = json.loads(config.read_text())
    assert value["lag"] == 7
    assert value["prompts_per_update"] * value["responses_per_prompt"] == 512
    assert not any(item is None for item in value.values())
    destination = tmp_path / "resolved"
    subprocess.run(command + ["build", str(config), str(destination)], check=True, capture_output=True)
    protocol = json.loads((destination / "protocol.json").read_text())
    assert protocol["warmup_updates"] == 7
    assert protocol["exact_staleness_updates"] == 993
    repeated = subprocess.run(command + ["init", str(config), "--lag", "32"], capture_output=True)
    assert repeated.returncode != 0
    assert json.loads(config.read_text())["lag"] == 7
