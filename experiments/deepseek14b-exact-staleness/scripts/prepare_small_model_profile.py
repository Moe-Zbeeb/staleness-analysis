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
QWEN_MODEL_ID = "Qwen/Qwen2.5-3B"
QWEN_MODEL_REVISION = "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
QWEN3_MODEL_ID = "Qwen/Qwen3-1.7B"
QWEN3_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
PROFILE_MODELS = {
    MODEL_ID: (MODEL_REVISION, "deepseek15b"),
    QWEN_MODEL_ID: (QWEN_MODEL_REVISION, "qwen25-3b"),
    QWEN3_MODEL_ID: (QWEN3_MODEL_REVISION, "qwen3-1p7b"),
}


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


def clone_source(source, destination, model_id=MODEL_ID, model_revision=MODEL_REVISION):
    if model_id not in PROFILE_MODELS or model_revision != PROFILE_MODELS[model_id][0]:
        raise ValueError("Unknown profiling model pin")
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
        ("MODEL_ID", BASE_MODEL_ID, model_id),
        ("MODEL_REVISION", BASE_MODEL_REVISION, model_revision),
    ):
        before = f'{key} = "{old}"'
        if replacement.count(before) != 1:
            raise ValueError(f"Unexpected baseline model pin: {key}")
        replacement = replacement.replace(before, f'{key} = "{new}"')
    init.write_text(replacement)
    expected_changes = ["src/deepseek_study/__init__.py"]
    if model_id in {QWEN_MODEL_ID, QWEN3_MODEL_ID}:
        assets = destination / "src/deepseek_study/dataset/assets.py"
        contents = assets.read_text()
        replacements = [
            (
                'if actual != expected_prompt or not loaded.decode(actual).endswith("<think>\\n"):',
                "if actual != expected_prompt:",
            ),
            ("Renderer changed the native DeepSeek thinking prompt", "Renderer changed the native Qwen prompt"),
            (
                "if (loaded.bos_token_id, loaded.eos_token_id) != (151646, 151643):",
                "if (loaded.bos_token_id, loaded.eos_token_id) != (original.bos_token_id, original.eos_token_id):",
            ),
            ("Unexpected DeepSeek BOS/EOS tokens", "Prepared tokenizer changed native Qwen BOS/EOS tokens"),
        ]
        for before, after in replacements:
            if contents.count(before) != 1:
                raise ValueError("Unexpected frozen tokenizer parity implementation")
            contents = contents.replace(before, after)
        assets.write_text(contents)
        expected_changes.append("src/deepseek_study/dataset/assets.py")
    (destination / "vendor").mkdir()
    (destination / "vendor/prime-rl").symlink_to((source / "vendor/prime-rl").resolve())
    changed = []
    for path in sorted((destination / "src").rglob("*.py")):
        name = str(path.relative_to(destination))
        if digest(path) != digest(source / name):
            changed.append(name)
    if changed != expected_changes:
        raise ValueError(f"Unexpected runtime source differences: {changed}")
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--model", choices=list(PROFILE_MODELS), default=MODEL_ID)
    parser.add_argument("--inference-gpus", type=int, choices=[3, 4, 5], default=4)
    args = parser.parse_args()
    source, directory = args.source.resolve(), args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(args.baseline.read_text())
    model_id = args.model
    model_revision, run_prefix = PROFILE_MODELS[model_id]
    release = directory / "release"
    changed = clone_source(source, release, model_id, model_revision)
    from huggingface_hub import snapshot_download

    model = directory / "assets/model"
    snapshot_download(
        model_id,
        revision=model_revision,
        local_dir=model,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE", "README.md"],
    )
    manifest = {
        "repo_id": model_id,
        "revision": model_revision,
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
            directory / f"{run_prefix}-exact256-timing-{os.environ.get('SLURM_JOB_ID', directory.parent.name)}"
        ),
        "metrics_mirror_root": str(args.mirror_root.resolve()),
        "inference_gpus": args.inference_gpus,
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
        "inference_gpus",
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
            "model_id": model_id,
            "model_revision": model_revision,
            "baseline": str(args.baseline.resolve()),
            "baseline_sha256": digest(args.baseline),
            "baseline_source": str(source),
            "baseline_package_sha256": digest(source / "PACKAGE_SHA256.json"),
            "changed_runtime_files": changed,
            "changed_runtime_values": ["MODEL_ID", "MODEL_REVISION"],
            "native_tokenizer_parity_adapted": model_id in {QWEN_MODEL_ID, QWEN3_MODEL_ID},
            "config_changes": config_changes,
            "same_question_membership_and_order": True,
            "included_questions": len(after_ids),
            "prime_rl_modified": False,
            "timing_only": True,
        },
    )
    print(json.dumps({"prepared": str(directory), "model_id": model_id}), flush=True)


if __name__ == "__main__":
    main()
