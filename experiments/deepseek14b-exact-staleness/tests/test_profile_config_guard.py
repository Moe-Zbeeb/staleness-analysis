import copy
import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).parents[1] / "scripts/compare_profile_replays.py"
spec = importlib.util.spec_from_file_location("compare_profile_replays", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def config():
    return {
        "output_dir": "baseline",
        "model": {"ac": {"mode": "full"}, "reshard_after_forward": True, "compile": None, "reduce_dtype": "float32"},
        "loss": {"kwargs": {"clip_epsilon": 0.2}},
        "optim": {"lr": 1e-6},
    }


def test_performance_change_requires_explicit_allowlist():
    left = config()
    right = copy.deepcopy(left)
    right["output_dir"] = "candidate"
    right["model"]["ac"] = None
    with pytest.raises(ValueError):
        module.compare_configs(left, right)
    assert module.compare_configs(left, right, ["ac"]) == {"ac": [{"mode": "full"}, None]}
    assert left == config()


@pytest.mark.parametrize("field", ["loss", "optim"])
def test_allowing_memory_setting_never_allows_algorithm_change(field):
    left = config()
    right = copy.deepcopy(left)
    right["model"]["ac"] = None
    right[field] = {}
    with pytest.raises(ValueError):
        module.compare_configs(left, right, ["ac"])


def test_precision_cannot_be_allowlisted():
    with pytest.raises(ValueError):
        module.compare_configs(config(), config(), ["reduce_dtype"])
