import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from benchmark_1p5b import run
from node_local_run import configure_local_git, copy_tree, digest, local_environment, secure_local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--matrix-results", type=Path, required=True)
    parser.add_argument("--local-runtime", action="store_true")
    args = parser.parse_args()
    workspace = secure_local(args.workspace)
    runtime = workspace / "runtime"
    python = runtime / "prime-rl/.venv/bin/python"
    root = workspace / ("pipeline-" + os.environ["SLURM_JOB_ID"])
    release = root / "release"
    if not args.local_runtime:
        if os.environ.get("SLURMD_NODENAME") != "deep-chungus-1":
            raise ValueError("The cached diagnostic runtime belongs to node 1")
        if root.exists() or args.shared.exists():
            raise FileExistsError("Pipeline benchmark needs fresh local and shared outputs")
        root.mkdir()
        copy_tree(workspace / "release", release, ("vendor", "__pycache__"))
        changes = {}
        replacements = {
            "src/deepseek_study/config.py": ("    trainer_reshard_after_forward: bool = True\n", ""),
            "src/deepseek_study/runtime/build.py": (
                '"reshard_after_forward": study.trainer_reshard_after_forward,',
                '"reshard_after_forward": True,',
            ),
        }
        package = json.loads((release / "PACKAGE_SHA256.json").read_text())
        for name, replacement in replacements.items():
            target = release / name
            source = args.control / Path(name).name
            if source.read_text().replace(*replacement) != target.read_text():
                raise ValueError(f"Unexpected adapter change beyond the resharding option: {name}")
            changes[name] = {"before": digest(target), "after": digest(source)}
            shutil.copy2(source, target)
            package[name] = digest(target)
        (release / "PACKAGE_SHA256.json").write_text(json.dumps(package, indent=2))
        (root / "source-changes.json").write_text(json.dumps(changes, indent=2))
        (release / "vendor").mkdir()
        (release / "vendor/prime-rl").symlink_to(runtime / "prime-rl", target_is_directory=True)
        configure_local_git(runtime)
        environment = local_environment(runtime, root, release)
        os.execve(python, [str(python), *sys.argv, "--local-runtime"], environment)

    from deepseek_study.config import StudyConfig
    from deepseek_study.dataset.assets import prepare
    from deepseek_study.runtime.launcher import verify_upstream

    verify_upstream(release)
    from compare_profile_replays import compare

    comparisons = {}
    for label, allowed in (
        ("baseline-repeat", ()),
        ("no-ac", ("ac",)),
        ("no-reshard", ("reshard_after_forward",)),
        ("compile", ("compile",)),
    ):
        comparisons[label] = compare(
            args.matrix_results / "trainer-baseline", args.matrix_results / ("trainer-" + label), 4, 3, allowed
        )
    comparison_text = json.dumps(comparisons, indent=2, allow_nan=False)
    (root / "matrix-comparisons.json").write_text(comparison_text)
    args.shared.mkdir(parents=True, exist_ok=False)
    (args.shared / "matrix-comparisons.json").write_text(comparison_text)
    baseline = json.loads(args.study.read_text())
    if baseline["lag"] != 256 or baseline["trainer_gpus"] != 4 or baseline["max_steps"] != 1000:
        raise ValueError("Unexpected production schedule or trainer topology")
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices) != 8 or len(set(devices)) != 8:
        raise ValueError("The pipeline diagnostic requires eight allocated GPUs")
    environment = {**os.environ, "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}
    os.chdir(root)
    run(
        [
            str(python),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=8",
            str(args.control / "gpu_health.py"),
            "--expected-gpus",
            "8",
            "--receipt",
            str(root / "health.json"),
        ],
        root / "health.log",
        environment,
        600,
    )
    assets = root / "assets"
    original = assets / "model"
    original.mkdir(parents=True)
    manifest = json.loads((release / "manifests/model.json").read_text())
    for record in manifest["files"]:
        name = record["name"]
        source = Path(baseline["model_path"]) / name if name == "tokenizer_config.json" else workspace / "model" / name
        target = original / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != record["sha256"]:
            raise ValueError(f"Original model artifact differs: {name}")
    for key, name in (("dataset_path", "train.parquet"), ("data_manifest", "manifest.json")):
        source = Path(baseline[key])
        shutil.copy2(source, assets / name)
        if digest(source) != digest(assets / name):
            raise ValueError("Dataset or prepared manifest changed during staging")
    prepare(original, assets / "train.parquet", assets / "native-model", release / "manifests/model.json")
    receipt = {
        "job_id": os.environ["SLURM_JOB_ID"],
        "node": os.environ["SLURMD_NODENAME"],
        "production_inference_gpus": baseline["inference_gpus"],
        "diagnostic_inference_gpus": 4,
        "bounded_updates_per_case": 2,
        "max_seconds_per_case": 2400,
        "production_modified": False,
        "cases": {},
    }
    try:
        for label, sequences, concurrency, reshard in (("baseline", 16, 64, True), ("candidate", 64, 256, False)):
            folder = root / label
            folder.mkdir()
            (folder / "release").symlink_to(release, target_is_directory=True)
            values = baseline | {
                "model_path": str(original),
                "prepared_model_path": str(assets / "native-model"),
                "dataset_path": str(assets / "train.parquet"),
                "data_manifest": str(assets / "manifest.json"),
                "output_dir": str(folder / "run"),
                "metrics_mirror_root": str(folder / "metrics"),
                "inference_gpus": 4,
                "inference_max_sequences": sequences,
                "rollout_concurrency": concurrency,
                "trainer_reshard_after_forward": reshard,
            }
            study = StudyConfig.model_validate(values)
            (folder / "study.json").write_text(study.model_dump_json(indent=2))
            run(
                [
                    str(python),
                    str(args.control / "run_bounded_profile.py"),
                    "--directory",
                    str(folder),
                    "--updates",
                    "2",
                    "--seconds",
                    "2400",
                ],
                folder / "supervisor.log",
                environment,
                2550,
            )
            report = json.loads((folder / "timing-summary.json").read_text())
            if report["status"] != "completed" or report["completed_updates"] != 2:
                raise RuntimeError("Pipeline diagnostic did not finish exactly two updates")
            receipt["cases"][label] = report
            (root / "summary.json").write_text(json.dumps(receipt, indent=2))
            args.shared.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["rsync", "-rt", "--exclude=release", "--exclude=assets", str(root) + "/", str(args.shared) + "/"],
                check=True,
                timeout=180,
            )
    finally:
        (root / "summary.json").write_text(json.dumps(receipt, indent=2))
        args.shared.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rsync", "-rt", "--exclude=release", "--exclude=assets", str(root) + "/", str(args.shared) + "/"],
            check=True,
            timeout=180,
        )


if __name__ == "__main__":
    main()
