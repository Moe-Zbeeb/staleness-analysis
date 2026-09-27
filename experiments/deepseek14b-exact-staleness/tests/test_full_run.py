import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load("prepare_full_run")
LAUNCH = load("launch_full_run")


def probes(count, failed=()):
    return {
        "allocated_devices": [str(i) for i in range(count)],
        "healthy_devices": [str(i) for i in range(count) if i not in failed],
        "results": [{"device": str(i), "healthy": i not in failed, "uuid": f"uuid-{i}"} for i in range(count)],
    }


def test_full_run_preserves_science_and_budget(tmp_path):
    baseline = json.loads((SCRIPTS.parent / "configs/exact256-80gb-seed42-v2.json").read_text())
    result = PREPARE.full_config(baseline, tmp_path / "full", 5)
    assert {k for k in baseline if baseline[k] != result[k]} == {"output_dir", "inference_gpus"}
    assert result["max_steps"] == 1000 and result["lag"] == 256 and result["checkpoint_interval"] == 100


@pytest.mark.parametrize("key,value", [("max_steps", 4), ("lag", 32), ("weight_decay", 0.01)])
def test_full_run_rejects_wrong_protocol(key, value, tmp_path):
    baseline = json.loads((SCRIPTS.parent / "configs/exact256-80gb-seed42-v2.json").read_text())
    baseline[key] = value
    with pytest.raises(ValueError, match="protocol"):
        PREPARE.full_config(baseline, tmp_path / "full", 5)


def test_all_nine_gpus_and_authorized_seven_gpu_fallback():
    assert LAUNCH.select_devices(probes(9), 9, 9) == [str(i) for i in range(9)]
    assert LAUNCH.select_devices(probes(8, {7}), 8, 7) == [str(i) for i in range(7)]


def test_occupied_gpu_is_not_a_hardware_fallback():
    value = probes(8, {7})
    value["results"][7]["error"] = "GPU is already occupied: 10% free"
    with pytest.raises(ValueError, match="occupied"):
        LAUNCH.select_devices(value, 8, 7)


def test_full_run_rejects_duplicate_gpu_aliases():
    value = probes(9)
    value["results"][8]["uuid"] = value["results"][7]["uuid"]
    with pytest.raises(ValueError, match="duplicated"):
        LAUNCH.select_devices(value, 9, 9)


def test_full_run_never_silently_discards_healthy_gpus():
    with pytest.raises(ValueError, match="topology"):
        LAUNCH.select_devices(probes(9), 9, 8)
