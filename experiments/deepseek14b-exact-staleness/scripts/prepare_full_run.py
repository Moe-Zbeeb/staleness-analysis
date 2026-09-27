import argparse
import hashlib
import json
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inference-gpus", type=int, required=True)
    parser.add_argument("--allocated-gpus", type=int, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--minimum-gpu-bytes", type=int, required=True)
    args = parser.parse_args()
    profile, control, output = args.profile.resolve(), args.control.resolve(), args.output.resolve()
    if output.exists() or (control / "full-run.json").exists():
        raise FileExistsError("Full run already prepared or output already exists")
    preparation = json.loads((profile / "preparation.json").read_text())
    model = preparation["model_id"]
    if (
        model not in PROFILE_MODELS
        or preparation["model_revision"] != PROFILE_MODELS[model][0]
        or not preparation["same_question_membership_and_order"]
        or preparation["prime_rl_modified"]
    ):
        raise ValueError("Profile model or data provenance is incompatible")
    release = profile / "release"
    for name, expected in json.loads((release / "PACKAGE_SHA256.json").read_text()).items():
        if digest(release / name) != expected:
            raise ValueError(f"Frozen profile artifact changed: {name}")
    sys.path.insert(0, str(release / "src"))
    from deepseek_study import MODEL_ID, MODEL_REVISION
    from deepseek_study.config import StudyConfig
    from deepseek_study.runtime.identity import capture, read_identity

    if (MODEL_ID, MODEL_REVISION) != (model, preparation["model_revision"]):
        raise ValueError("Imported model differs from the profile manifest")
    baseline = StudyConfig.read(profile / "study.json")
    identity = read_identity(baseline.output_dir / "source/identity.json")
    if capture(release, baseline)["sha256"] != identity["sha256"]:
        raise ValueError("Profile source, data or runtime changed")
    study = StudyConfig.model_validate(full_config(baseline.model_dump(mode="json"), output, args.inference_gpus))
    participating = study.trainer_gpus + study.inference_gpus
    if not (args.allocated_gpus == participating or (args.allocated_gpus, participating) == (8, 7)):
        raise ValueError("Every GPU must participate except the authorized seven-GPU fallback")
    control.mkdir(parents=True, exist_ok=True)
    write(control / "study.json", study.model_dump(mode="json"))
    write(
        control / "full-run.json",
        {
            "mode": "full_training", "model_id": model, "model_revision": MODEL_REVISION,
            "profile": str(profile), "release": str(release), "node": args.node,
            "allocated_gpus": args.allocated_gpus, "minimum_gpu_bytes": args.minimum_gpu_bytes,
            "identity_sha256": identity["sha256"], "config_sha256": study.fingerprint(),
            "study_sha256": digest(control / "study.json"),
            "config_changes": {
                key: {"before": baseline.model_dump(mode="json")[key], "after": value}
                for key, value in study.model_dump(mode="json").items()
                if baseline.model_dump(mode="json")[key] != value
            },
            "scripts_sha256": {
                name: digest(control / name)
                for name in ("launch_full_run.py", "full_run_job.sh", "probe_allocated_gpus.py")
            },
            "starts_from_initial_model": True, "profile_updates_reused": 0,
            "checkpoint_interval": study.checkpoint_interval, "maximum_updates": study.max_steps,
            "lag": study.lag, "bounded_profile_supervisor": False,
        },
    )
    print(json.dumps({"prepared": str(control), "output": str(output), "config_sha256": study.fingerprint()}))


if __name__ == "__main__":
    main()
