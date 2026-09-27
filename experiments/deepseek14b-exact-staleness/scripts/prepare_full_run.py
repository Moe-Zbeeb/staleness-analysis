import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from prepare_small_model_profile import PROFILE_MODELS


def full_config(baseline, output, inference_gpus):
    required = {"max_steps": 1000, "lag": 256, "checkpoint_interval": 100, "trainer_gpus": 4, "weight_decay": 0.0}
    if any(baseline.get(key) != value for key, value in required.items()):
        raise ValueError("Profile does not match the authorized full-run protocol")
    if inference_gpus not in {3, 4, 5}:
        raise ValueError("Unsupported inference layout")
    return {**baseline, "output_dir": str(output), "inference_gpus": inference_gpus}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def allowed_nodes(values):
    nodes = list(dict.fromkeys(node.strip() for value in values for node in value.split(",")))
    if not nodes or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", node) for node in nodes):
        raise ValueError("Provide explicit candidate node names")
    return nodes


def fresh_config(baseline, output, inference_gpus, allocated_gpus, minimum_gpu_bytes):
    result = full_config(baseline, output, inference_gpus)
    if allocated_gpus != 8 or inference_gpus not in {3, 4} or minimum_gpu_bytes < 79_000_000_000:
        raise ValueError(
            "Fresh studies require an exclusive eight-A100-80GB allocation with seven or eight healthy GPUs"
        )
    return result


def control_scripts(control):
    required = ("launch_full_run.py", "full_run_job.sh", "probe_allocated_gpus.py")
    optional = (
        "prepare_full_run.py",
        "prepare_small_model_profile.py",
        "prepare_cached_full_run.py",
        "cached_full_run_job.sh",
        "prelaunch.json",
        "preparation-input.json",
    )
    permitted = {
        *required,
        *optional,
        "preparation.json",
        "device-probes-preparation.json",
        "import-warmup-preparation.json",
    }
    if control.exists():
        for path in control.iterdir():
            if path.name == "work" and path.is_dir() and not path.is_symlink():
                continue
            if (
                path.name == "__pycache__"
                and path.is_dir()
                and not path.is_symlink()
                and all(item.is_file() and not item.is_symlink() and item.suffix == ".pyc" for item in path.iterdir())
            ):
                continue
            if not path.is_file() or path.name not in permitted:
                raise FileExistsError("Launch control contains unexpected artifacts or an existing run")
    return {
        name: digest(control / name)
        for name in (*required, *optional)
        if name in required or (control / name).is_file()
    }


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--profile", type=Path)
    source.add_argument("--study", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inference-gpus", type=int, required=True)
    parser.add_argument("--allocated-gpus", type=int, required=True)
    parser.add_argument("--node", action="append", required=True)
    parser.add_argument("--minimum-gpu-bytes", type=int, required=True)
    args = parser.parse_args()
    if bool(args.study) != bool(args.release):
        parser.error("--release is required with --study and must not be used with --profile")
    control, output = args.control.resolve(), args.output.resolve()
    nodes = allowed_nodes(args.node)
    if output.exists() or (control / "full-run.json").exists():
        raise FileExistsError("Full run already prepared or output already exists")
    scripts_sha256 = control_scripts(control)
    from launch_full_run import startup_deadlines

    profile = args.profile.resolve() if args.profile else None
    release = profile / "release" if profile else args.release.resolve()
    for name, expected in json.loads((release / "PACKAGE_SHA256.json").read_text()).items():
        if digest(release / name) != expected:
            raise ValueError(f"Frozen profile artifact changed: {name}")
    sys.path.insert(0, str(release / "src"))
    from deepseek_study import MODEL_ID, MODEL_REVISION
    from deepseek_study.config import StudyConfig
    from deepseek_study.dataset.assets import validate_prepared
    from deepseek_study.runtime.identity import capture, read_identity

    if MODEL_ID not in PROFILE_MODELS or MODEL_REVISION != PROFILE_MODELS[MODEL_ID][0]:
        raise ValueError("Imported model differs from the pinned study models")
    preflight = None
    if profile:
        preparation = json.loads((profile / "preparation.json").read_text())
        if (
            (MODEL_ID, MODEL_REVISION) != (preparation["model_id"], preparation["model_revision"])
            or not preparation["same_question_membership_and_order"]
            or preparation["prime_rl_modified"]
        ):
            raise ValueError("Profile model or data provenance is incompatible")
        baseline = StudyConfig.read(profile / "study.json")
        identity = read_identity(baseline.output_dir / "source/identity.json")
        if capture(release, baseline)["sha256"] != identity["sha256"]:
            raise ValueError("Profile source, data or runtime changed")
        values = full_config(baseline.model_dump(mode="json"), output, args.inference_gpus)
        study = StudyConfig.model_validate(values)
    else:
        baseline = StudyConfig.read(args.study)
        values = fresh_config(
            baseline.model_dump(mode="json"),
            output,
            args.inference_gpus,
            args.allocated_gpus,
            args.minimum_gpu_bytes,
        )
        study = StudyConfig.model_validate(values)
        preflight = validate_prepared(study)
        identity = capture(release, study)
    participating = study.trainer_gpus + study.inference_gpus
    if not (args.allocated_gpus == participating or (args.allocated_gpus, participating) == (8, 7)):
        raise ValueError("Every GPU must participate except the authorized seven-GPU fallback")
    control.mkdir(parents=True, exist_ok=True)
    write(control / "study.json", study.model_dump(mode="json"))
    write(
        control / "full-run.json",
        {
            "mode": "full_training",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "preparation_mode": "verified_profile" if profile else "fresh_study",
            "profile": str(profile) if profile else None,
            "study_source": str(args.study.resolve()) if args.study else str(profile / "study.json"),
            "release": str(release),
            "allowed_nodes": nodes,
            **({"node": nodes[0]} if len(nodes) == 1 else {}),
            "allocated_gpus": args.allocated_gpus,
            "minimum_gpu_bytes": args.minimum_gpu_bytes,
            "identity_sha256": identity["sha256"],
            "config_sha256": study.fingerprint(),
            "study_sha256": digest(control / "study.json"),
            "config_changes": {
                key: {"before": baseline.model_dump(mode="json")[key], "after": value}
                for key, value in study.model_dump(mode="json").items()
                if baseline.model_dump(mode="json")[key] != value
            },
            "scripts_sha256": scripts_sha256,
            "startup_deadlines": startup_deadlines(),
            "asset_preflight": preflight,
            "starts_from_initial_model": True,
            "profile_updates_reused": 0,
            "checkpoint_interval": study.checkpoint_interval,
            "maximum_updates": study.max_steps,
            "lag": study.lag,
            "bounded_profile_supervisor": False,
        },
    )
    print(json.dumps({"prepared": str(control), "output": str(output), "config_sha256": study.fingerprint()}))


if __name__ == "__main__":
    main()
