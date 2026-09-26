import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
BASE_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
BASE_MODEL_REVISION = "1df8507178afcc1bef68cd8c393f61a886323761"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def clone_source(source, destination):
    checksums = json.loads((source / "PACKAGE_SHA256.json").read_text())
    for name, expected in checksums.items():
        if digest(source / name) != expected:
            raise ValueError(f"Frozen baseline source changed: {name}")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("src", "scripts"):
        shutil.copytree(source / name, destination / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(source / "pyproject.toml", destination / "pyproject.toml")
    init = destination / "src/deepseek_study/__init__.py"
    original = init.read_text()
    replacement = original
    for key, old, new in (
        ("MODEL_ID", BASE_MODEL_ID, MODEL_ID),
        ("MODEL_REVISION", BASE_MODEL_REVISION, MODEL_REVISION),
    ):
        before = f'{key} = "{old}"'
        if replacement.count(before) != 1:
            raise ValueError(f"Unexpected baseline model pin: {key}")
        replacement = replacement.replace(before, f'{key} = "{new}"')
    init.write_text(replacement)
    (destination / "vendor").mkdir()
    (destination / "vendor/prime-rl").symlink_to((source / "vendor/prime-rl").resolve())
    changed = []
    for path in sorted((destination / "src").rglob("*.py")):
        name = str(path.relative_to(destination))
        if digest(path) != digest(source / name):
            changed.append(name)
    if changed != ["src/deepseek_study/__init__.py"]:
        raise ValueError(f"Unexpected runtime source differences: {changed}")
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--mirror-root", type=Path, required=True)
    args = parser.parse_args()
    source, directory = args.source.resolve(), args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(args.baseline.read_text())
    release = directory / "release"
    changed = clone_source(source, release)
    from huggingface_hub import snapshot_download

    model = directory / "assets/model"
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=model,
        allow_patterns=["*.json", "*.safetensors", "LICENSE", "README.md"],
    )
    manifest = {
        "repo_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "path": str(model),
        "files": [
            {"name": path.name, "size": path.stat().st_size, "sha256": digest(path)}
            for path in sorted(model.iterdir())
            if path.is_file() and not path.name.startswith(".")
        ],
    }
    write(release / "manifests/model.json", manifest)
    study = {
        **baseline,
        "model_path": str(model),
        "prepared_model_path": str(directory / "assets/native-model"),
        "data_manifest": str(directory / "assets/train-manifest.json"),
        "output_dir": str(
            directory / f"deepseek15b-exact256-timing-{os.environ.get('SLURM_JOB_ID', directory.parent.name)}"
        ),
        "metrics_mirror_root": str(args.mirror_root.resolve()),
    }
    config_changes = {
        key: {"before": baseline[key], "after": value} for key, value in study.items() if baseline[key] != value
    }
    if set(config_changes) - {
        "model_path",
        "prepared_model_path",
        "data_manifest",
        "output_dir",
        "metrics_mirror_root",
    }:
        raise ValueError("Timing profile changed a training setting")
    write(directory / "study.json", study)
    environment = {**os.environ, "PYTHONPATH": str(release / "src")}
    cli = [sys.executable, "-m", "deepseek_study.cli"]
    subprocess.run(
        cli
        + [
            "prepare",
            "--model",
            str(model),
            "--dataset",
            study["dataset_path"],
            "--destination",
            study["prepared_model_path"],
            "--manifest",
            str(release / "manifests/model.json"),
        ],
        env=environment,
        check=True,
    )
    subprocess.run(cli + ["prepare-data", str(directory / "study.json")], env=environment, check=True)
    before = json.loads(Path(baseline["data_manifest"]).read_text())
    after = json.loads(Path(study["data_manifest"]).read_text())
    before_ids = [record["id"] for record in before["records"] if record["included"]]
    after_ids = [record["id"] for record in after["records"] if record["included"]]
    if before_ids != after_ids:
        raise ValueError("Prepared question membership differs from the 14B baseline")
    subprocess.run(cli + ["check", str(directory / "study.json")], env=environment, check=True)
    files = [path for path in release.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    write(release / "PACKAGE_SHA256.json", {str(path.relative_to(release)): digest(path) for path in sorted(files)})
    write(
        directory / "preparation.json",
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "baseline": str(args.baseline.resolve()),
            "baseline_sha256": digest(args.baseline),
            "baseline_source": str(source),
            "baseline_package_sha256": digest(source / "PACKAGE_SHA256.json"),
            "changed_runtime_files": changed,
            "changed_runtime_values": ["MODEL_ID", "MODEL_REVISION"],
            "config_changes": config_changes,
            "same_question_membership_and_order": True,
            "included_questions": len(after_ids),
            "prime_rl_modified": False,
            "timing_only": True,
        },
    )
    print(json.dumps({"prepared": str(directory), "model_id": MODEL_ID}), flush=True)


if __name__ == "__main__":
    main()
