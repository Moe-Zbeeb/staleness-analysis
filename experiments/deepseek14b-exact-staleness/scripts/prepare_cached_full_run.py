import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


CONTROL_FILES = {
    "prepare_cached_full_run.py",
    "prepare_full_run.py",
    "prepare_small_model_profile.py",
    "launch_full_run.py",
    "probe_allocated_gpus.py",
    "full_run_job.sh",
    "cached_full_run_job.sh",
    "preparation-input.json",
}
SPEC_FIELDS = {
    "source",
    "old_profile",
    "model_id",
    "model_revision",
    "root",
    "output",
    "allowed_nodes",
    "allocated_gpus",
    "min_gpu_bytes",
    "allow_seven_healthy",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def verify_prelaunch(control):
    inventory = json.loads((control / "prelaunch.json").read_text())
    if inventory.get("schema_version") != 1 or set(inventory.get("files_sha256", {})) != CONTROL_FILES:
        raise ValueError("Prelaunch inventory must cover exactly the authorized control files")
    for name, expected in inventory["files_sha256"].items():
        path = control / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Prelaunch artifact changed: {name}")
    specification = json.loads((control / "preparation-input.json").read_text())
    if set(specification) != SPEC_FIELDS:
        raise ValueError("Cached preparation specification has missing or unexpected fields")
    return specification


def validate_specification(specification, environment):
    from prepare_small_model_profile import QWEN_MODEL_ID, QWEN_MODEL_REVISION

    if (specification["model_id"], specification["model_revision"]) != (QWEN_MODEL_ID, QWEN_MODEL_REVISION):
        raise ValueError("Cached preparation requires the authorized pinned Qwen2.5-3B model")
    if (
        specification["allocated_gpus"] != 8
        or type(specification["min_gpu_bytes"]) is not int
        or specification["min_gpu_bytes"] < 79_000_000_000
        or type(specification["allow_seven_healthy"]) is not bool
    ):
        raise ValueError("Cached preparation requires an eight-A100-80GB allocation")
    nodes = specification["allowed_nodes"]
    if (
        not isinstance(nodes, list)
        or not nodes
        or len(nodes) != len(set(nodes))
        or any(not isinstance(node, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", node) for node in nodes)
    ):
        raise ValueError("Cached preparation requires explicit allowed node names")
    node = environment.get("SLURMD_NODENAME")
    if node not in nodes or not environment.get("SLURM_JOB_ID") or int(environment.get("SLURM_RESTART_COUNT", "0")):
        raise ValueError("Unexpected node, missing allocation or unsafe automatic restart")
    devices = environment.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 8 or len(set(devices)) != 8 or not all(devices):
        raise ValueError("Cached preparation requires eight distinct allocated devices")
    for key in ("source", "old_profile", "root", "output"):
        if not isinstance(specification[key], str) or not Path(specification[key]).is_absolute():
            raise ValueError("Cached preparation paths must be absolute")
    return node, devices


def choose_topology(probes, specification, allocated):
    from launch_full_run import select_devices

    if probes["allocated_devices"] != allocated:
        raise ValueError("Preparation probe differs from the current allocation")
    count = len(probes["healthy_devices"])
    if count != 8 and not (count == 7 and specification["allow_seven_healthy"]):
        raise ValueError("GPU health does not match the authorized full or seven-GPU topology")
    visible = select_devices(probes, specification["allocated_gpus"], count)
    records = {row["device"]: row for row in probes["results"]}
    if any(
        "A100" not in records[device]["name"] or records[device]["bytes"] < specification["min_gpu_bytes"]
        for device in visible
    ):
        raise ValueError("Preparation requires healthy A100-80GB devices")
    return visible, count - 4


def manifest_ids(path):
    manifest = json.loads(path.read_text())
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest.get("format") != 1 or manifest.get("sha256") != expected:
        raise ValueError("Prepared data manifest failed integrity validation")
    records = manifest["records"]
    identifiers = [record["id"] for record in records if record["included"]]
    if (
        not identifiers
        or len(set(record["id"] for record in records)) != len(records)
        or len(records) != manifest["source_rows"]
        or len(identifiers) != manifest["included_rows"]
        or len(records) - len(identifiers) != manifest["excluded_rows"]
    ):
        raise ValueError("Prepared data manifest has inconsistent question accounting")
    return identifiers, manifest


def verify_membership(reference, prepared):
    before, source = manifest_ids(reference)
    after, target = manifest_ids(prepared)
    if before != after:
        raise ValueError("Prepared question membership or order differs from the v3 baseline")
    for key in ("source_sha256", "reward", "prompt_instruction", "prompt_max_tokens", "reward_timeout_seconds"):
        if source["contract"][key] != target["contract"][key]:
            raise ValueError("Prepared dataset or grading contract differs from the v3 baseline")
    return len(after)


def freeze_release(release):
    files = [release / "pyproject.toml", release / "manifests/model.json"]
    for name in ("src", "scripts"):
        files.extend(
            path
            for path in (release / name).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    if any(path.is_symlink() for path in files):
        raise ValueError("Authored release files must not be symlinks")
    checksums = {str(path.relative_to(release)): digest(path) for path in sorted(files)}
    write(release / "PACKAGE_SHA256.json", checksums)
    return checksums


def run_stage(name, command, environment, timeout, cwd):
    print(json.dumps({"stage": name, "status": "starting", "timeout_seconds": timeout}), flush=True)
    started = time.monotonic()
    subprocess.run(command, env=environment, cwd=cwd, check=True, timeout=timeout)
    print(json.dumps({"stage": name, "status": "complete", "seconds": time.monotonic() - started}), flush=True)


def prepare(control):
    control = control.resolve()
    specification = verify_prelaunch(control)
    from launch_full_run import GPU_PROBE_PARENT_MARGIN_SECONDS, GPU_PROBE_TIMEOUT_SECONDS

    node, allocated = validate_specification(specification, os.environ)
    source = Path(specification["source"]).resolve()
    old_profile = Path(specification["old_profile"]).resolve()
    output = Path(specification["output"]).resolve()
    work = control / "work"
    if output.exists() or work.exists() or (control / "full-run.json").exists():
        raise FileExistsError("Refusing to overwrite or repeat cached full-run preparation")
    if output.is_relative_to(control) or control.is_relative_to(output):
        raise ValueError("Training output and launch control must be separate")
    environment = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    run_stage(
        "probe_all_allocated_gpus",
        [
            sys.executable,
            str(control / "probe_allocated_gpus.py"),
            "--output",
            str(control / "device-probes-preparation.json"),
            "--timeout",
            str(GPU_PROBE_TIMEOUT_SECONDS),
        ],
        environment,
        GPU_PROBE_TIMEOUT_SECONDS + GPU_PROBE_PARENT_MARGIN_SECONDS,
        control,
    )
    probes = json.loads((control / "device-probes-preparation.json").read_text())
    visible, inference_gpus = choose_topology(probes, specification, allocated)
    from prepare_small_model_profile import clone_source

    print(json.dumps({"stage": "clone_verified_source", "status": "starting"}), flush=True)
    work.mkdir()
    release = work / "release"
    changed = clone_source(source, release, specification["model_id"], specification["model_revision"])
    model_manifest = old_profile / "release/manifests/model.json"
    model = old_profile / "assets/model"
    metadata = json.loads(model_manifest.read_text())
    if (metadata["repo_id"], metadata["revision"]) != (specification["model_id"], specification["model_revision"]):
        raise ValueError("Cached model manifest differs from the authorized model pin")
    (release / "manifests").mkdir()
    shutil.copyfile(model_manifest, release / "manifests/model.json")
    environment["PYTHONPATH"] = str(release / "src")
    cli = [sys.executable, "-m", "deepseek_study.cli"]
    study_path = work / "study-input.json"
    run_stage(
        "initialize_full_study",
        cli
        + [
            "init",
            str(study_path),
            "--lag",
            "256",
            "--max-steps",
            "1000",
            "--profile",
            "80gb",
            "--seed",
            "42",
            "--root",
            specification["root"],
        ],
        environment,
        900,
        release,
    )
    baseline = json.loads(study_path.read_text())
    values = {
        **baseline,
        "model_path": str(model),
        "prepared_model_path": str(work / "assets/native-model"),
        "data_manifest": str(work / "assets/train-manifest-v3.json"),
        "output_dir": str(output),
        "inference_gpus": inference_gpus,
        "reasoning_required": False,
    }
    study_path.write_text(json.dumps(values, indent=2) + "\n")
    run_stage(
        "verify_cached_model_and_prepare_native_tokenizer",
        cli
        + [
            "prepare",
            "--model",
            str(model),
            "--dataset",
            values["dataset_path"],
            "--destination",
            values["prepared_model_path"],
            "--manifest",
            str(release / "manifests/model.json"),
        ],
        environment,
        1800,
        release,
    )
    run_stage("prepare_reference_manifest", cli + ["prepare-data", str(study_path)], environment, 1800, release)
    count = verify_membership(source / "assets/train-manifest-v3.json", Path(values["data_manifest"]))
    run_stage("validate_study", cli + ["check", str(study_path)], environment, 900, release)
    checksums = freeze_release(release)
    write(
        control / "preparation.json",
        {
            "model_id": specification["model_id"],
            "model_revision": specification["model_revision"],
            "baseline_source": str(source),
            "baseline_package_sha256": digest(source / "PACKAGE_SHA256.json"),
            "model_manifest_sha256": digest(model_manifest),
            "model_reused_from": str(model),
            "release": str(release),
            "frozen_files": len(checksums),
            "changed_runtime_files": changed,
            "node": node,
            "allocated_devices": allocated,
            "healthy_devices": visible,
            "trainer_gpus": 4,
            "inference_gpus": inference_gpus,
            "same_question_membership_and_order": True,
            "included_questions": count,
            "prime_rl_modified": False,
            "starts_from_initial_model": True,
            "profile_updates_reused": 0,
            "downloads": False,
            "config_changes": {
                key: {"before": baseline[key], "after": value}
                for key, value in values.items()
                if baseline[key] != value
            },
        },
    )
    run_stage(
        "prepare_immutable_launch",
        [
            sys.executable,
            str(control / "prepare_full_run.py"),
            "--study",
            str(study_path),
            "--release",
            str(release),
            "--control",
            str(control),
            "--output",
            str(output),
            "--inference-gpus",
            str(inference_gpus),
            "--allocated-gpus",
            "8",
            "--node",
            node,
            "--minimum-gpu-bytes",
            str(specification["min_gpu_bytes"]),
        ],
        environment,
        900,
        control,
    )
    return [sys.executable, str(control / "launch_full_run.py"), "--control", str(control)], environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    command, environment = prepare(args.control)
    print(json.dumps({"stage": "launch_full_training", "status": "starting"}), flush=True)
    os.execve(sys.executable, command, environment)


if __name__ == "__main__":
    main()
