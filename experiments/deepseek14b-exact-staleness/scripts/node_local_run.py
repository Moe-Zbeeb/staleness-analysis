import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from local_backup import atomic_json, digest


PATH_FIELDS = {
    "model_path",
    "dataset_path",
    "data_manifest",
    "prepared_model_path",
    "output_dir",
    "metrics_mirror_root",
}


def relocate_study(baseline, workspace, name):
    workspace = Path(workspace)
    result = dict(baseline)
    result.update(
        model_path=str(workspace / "assets/model"),
        dataset_path=str(workspace / "assets/train.parquet"),
        data_manifest=str(workspace / "assets/train-manifest.json"),
        prepared_model_path=str(workspace / "assets/native-model"),
        output_dir=str(workspace / "runs" / name),
        metrics_mirror_root=None,
    )
    if {k: v for k, v in result.items() if k not in PATH_FIELDS} != {
        k: v for k, v in baseline.items() if k not in PATH_FIELDS
    }:
        raise ValueError("Storage relocation changed the scientific configuration")
    return result


def local_environment(runtime, workspace, release):
    cache = Path(workspace) / "cache"
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith("RUNBOARD_") or key in {"PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"}:
            environment.pop(key)
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(Path(release) / "src"),
            "VIRTUAL_ENV": str(Path(runtime) / "prime-rl/.venv"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "DEEPSEEK_STUDY_RUNBOARD": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "TOKENIZERS_PARALLELISM": "false",
            "TMPDIR": str(cache / "tmp"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
            "HF_HOME": str(cache / "huggingface"),
            "CUDA_CACHE_PATH": str(cache / "cuda"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
            "TORCH_EXTENSIONS_DIR": str(cache / "extensions"),
            "UV_CACHE_DIR": str(cache / "uv"),
            "VLLM_CACHE_ROOT": str(cache / "vllm"),
            "PATH": str(Path(runtime) / "prime-rl/.venv/bin") + os.pathsep + os.environ.get("PATH", ""),
        }
    )
    for key in (
        "TMPDIR",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "CUDA_CACHE_PATH",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "UV_CACHE_DIR",
        "VLLM_CACHE_ROOT",
    ):
        Path(environment[key]).mkdir(parents=True, exist_ok=True)
    return environment


def command(args, **kwargs):
    print(json.dumps({"stage": "command", "args": [str(a) for a in args]}), flush=True)
    return subprocess.run([str(a) for a in args], check=True, **kwargs)


def copy_tree(source, destination, exclude=()):
    destination.mkdir(parents=True, exist_ok=True)
    command(["rsync", "-a", *["--exclude=" + x for x in exclude], str(source) + "/", str(destination) + "/"])


def copy_environment(source, destination):
    copy_tree(source, destination, ("site-packages",))
    relative = Path("lib/python3.12/site-packages")
    target = destination / relative
    target.mkdir(parents=True, exist_ok=True)
    entries = list((source / relative).iterdir())

    def transfer(entry):
        subprocess.run(
            [
                "rsync",
                "-a",
                "--exclude=tests",
                "--exclude=__pycache__",
                "--exclude=*.pyc",
                str(entry),
                str(target) + "/",
            ],
            check=True,
            capture_output=True,
            timeout=1800,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(transfer, entry) for entry in entries]
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            if completed % 25 == 0 or completed == len(entries):
                print(
                    json.dumps({"stage": "stage_runtime", "copied_entries": completed, "total": len(entries)}),
                    flush=True,
                )


def secure_local(path):
    path = Path(path)
    if not path.is_absolute() or not path.is_relative_to("/tmp") or path.is_symlink():
        raise ValueError("Working roots must be absolute node-local /tmp paths")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.stat().st_uid != os.getuid() or path.resolve() != path:
        raise ValueError("Working root has unsafe ownership or symlink parents")
    os.chmod(path, 0o700)
    mount = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(path)], text=True).strip()
    if mount not in {"zfs", "ext4", "xfs", "btrfs"}:
        raise ValueError(f"Working root is not on a supported local disk: {mount}")
    return path


def relocate_runtime(runtime, shared_prime, release):
    local_prime = runtime / "prime-rl"
    venv = local_prime / ".venv"
    shared_python = (shared_prime / ".venv/bin/python").resolve().parent.parent
    prestaged = runtime / "PRESTAGED_RUNTIME.json"
    if prestaged.is_file():
        if json.loads(prestaged.read_text())["source"] != str(shared_prime):
            raise ValueError("Pre-staged runtime has a different source")
    else:
        copy_tree(shared_prime, local_prime, (".venv", ".cache", "outputs", "wandb", "__pycache__"))
        copy_tree(shared_python, runtime / "python")
        copy_environment(shared_prime / ".venv", venv)
    changed = {}
    cfg = venv / "pyvenv.cfg"
    before = cfg.read_text()
    lines = ["home = " + str(runtime / "python/bin") if x.startswith("home = ") else x for x in before.splitlines()]
    cfg.write_text("\n".join(lines) + "\n")
    changed[str(cfg)] = {"before_sha256": hashlib.sha256(before.encode()).hexdigest(), "after_sha256": digest(cfg)}
    for name in ("python", "python3", "python3.12"):
        path = venv / "bin" / name
        if path.is_symlink() or path.exists():
            path.unlink()
        path.symlink_to(runtime / "python/bin/python3.12")
    for path in (venv / "bin").iterdir():
        if path.is_symlink() or not path.is_file():
            continue
        with path.open("rb") as stream:
            prefix = stream.read(4096)
        if not prefix.startswith(b"#!"):
            continue
        before = path.read_text()
        after = before.replace(str(shared_prime / ".venv"), str(venv))
        if after != before:
            path.write_text(after)
            changed[str(path)] = {
                "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
                "after_sha256": digest(path),
            }
    site = venv / "lib/python3.12/site-packages"
    for path in site.glob("*.pth"):
        before = path.read_text()
        after = before.replace(str(shared_prime), str(local_prime))
        if path.name == "_editable_impl_deepseek_staleness_study.pth":
            after = str(release / "src") + "\n"
        if "/mnt/" in after:
            raise ValueError(f"Editable dependency still references shared storage: {path}")
        if after != before:
            path.write_text(after)
            changed[str(path)] = {
                "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
                "after_sha256": digest(path),
            }
    for path in venv.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(runtime):
            raise ValueError(f"Runtime symlink escapes local storage: {path}")
    atomic_json(runtime / "relocation.json", {"source": str(shared_prime), "changed_launch_metadata": changed})
    return venv / "bin/python"


def stage(spec, control):
    workspace = secure_local(spec["workspace"])
    runtime = secure_local(spec["runtime"])
    if shutil.disk_usage(workspace).free < 150 * 1024**3:
        raise RuntimeError("At least 150 GiB free local space is required before staging")
    if (workspace / "ready.json").exists():
        raise FileExistsError("Local run is already prepared")
    baseline = json.loads(Path(spec["baseline_study"]).read_text())
    release = workspace / "release"
    source = Path(spec["release"])
    if release.exists():
        raise FileExistsError("Local release already exists; inspect incomplete staging")
    for name, expected in json.loads((source / "PACKAGE_SHA256.json").read_text()).items():
        if digest(source / name) != expected:
            raise ValueError(f"Frozen source differs: {name}")
    copy_tree(source, release, ("vendor", "__pycache__"))
    (release / "vendor").mkdir()
    (release / "vendor/prime-rl").symlink_to(runtime / "prime-rl")
    python = relocate_runtime(runtime, Path(spec["shared_prime"]), release)
    env = local_environment(runtime, workspace, release)
    assets = workspace / "assets"
    assets.mkdir()
    model = assets / "model"
    model.mkdir()
    model_manifest = json.loads((release / "manifests/model.json").read_text())
    for item in model_manifest["files"]:
        print(json.dumps({"stage": "stage_model_file", "file": item["name"], "bytes": item["size"]}), flush=True)
        original = Path(baseline["model_path"]) / item["name"]
        target = model / item["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if target.stat().st_size != item["size"] or digest(target) != item["sha256"]:
            raise ValueError(f"Local model checksum mismatch: {item['name']}")
    for field, name in (("dataset_path", "train.parquet"), ("data_manifest", "train-manifest.json")):
        original, target = Path(baseline[field]), assets / name
        shutil.copy2(original, target)
        if digest(original) != digest(target):
            raise ValueError(f"Local asset checksum mismatch: {field}")
    values = relocate_study(baseline, workspace, spec["run_name"])
    atomic_json(workspace / "study.json", values)
    command(
        [
            python,
            "-m",
            "deepseek_study.cli",
            "prepare",
            "--model",
            model,
            "--dataset",
            assets / "train.parquet",
            "--destination",
            assets / "native-model",
            "--manifest",
            release / "manifests/model.json",
        ],
        env=env,
        cwd=release,
        timeout=1800,
    )
    command([python, "-m", "deepseek_study.cli", "check", workspace / "study.json"], env=env, cwd=release, timeout=900)
    command(
        [
            python,
            "-c",
            "import sys, pathlib, prime_rl, torch, vllm, verifiers; "
            "root=pathlib.Path(sys.argv[1]); "
            "assert pathlib.Path(sys.base_prefix).is_relative_to(root); "
            "assert all(pathlib.Path(m.__file__).resolve().is_relative_to(root) for m in (prime_rl,torch,vllm,verifiers)); "
            "print('Runtime imports and CPython base are node-local')",
            runtime,
        ],
        env=env,
        cwd=release,
        timeout=180,
    )
    old_tokenizer = Path(baseline["prepared_model_path"]) / "tokenizer_config.json"
    if digest(old_tokenizer) != digest(assets / "native-model/tokenizer_config.json"):
        raise ValueError("Storage migration changed the prepared tokenizer")
    old_receipt = json.loads((Path(baseline["prepared_model_path"]) / "study-assets.json").read_text())
    new_receipt = json.loads((assets / "native-model/study-assets.json").read_text())
    if new_receipt["dataset_sha256"] != old_receipt["dataset_sha256"]:
        raise ValueError("Storage migration changed the dataset")
    for name in ("local_backup.py", "node_local_run.py", "launch_full_run.py", "probe_allocated_gpus.py"):
        shutil.copy2(control / name, workspace / name)
    atomic_json(workspace / "storage-spec.json", spec)
    receipt = {
        "node": os.environ["SLURMD_NODENAME"],
        "python": str(python),
        "source": str(source),
        "release": str(release),
        "study_sha256": digest(workspace / "study.json"),
        "dataset_sha256": new_receipt["dataset_sha256"],
        "baseline_study_sha256": digest(spec["baseline_study"]),
        "data_manifest_sha256": digest(assets / "train-manifest.json"),
        "scientific_configuration_unchanged": True,
        "prepared_tokenizer_sha256": digest(assets / "native-model/tokenizer_config.json"),
        "package_sha256": digest(release / "PACKAGE_SHA256.json"),
        "control_sha256": {
            name: digest(workspace / name)
            for name in (
                "local_backup.py",
                "node_local_run.py",
                "launch_full_run.py",
                "probe_allocated_gpus.py",
                "storage-spec.json",
            )
        },
    }
    atomic_json(workspace / "ready.json", receipt)
    atomic_json(control / "local-ready.json", receipt)
    return receipt


def run(spec):
    workspace, runtime = Path(spec["workspace"]), Path(spec["runtime"])
    receipt = json.loads((workspace / "ready.json").read_text())
    if os.environ.get("SLURMD_NODENAME") != receipt["node"] or int(os.environ.get("SLURM_RESTART_COUNT", "0")):
        raise ValueError("Wrong node or unsupported automatic restart")
    if digest(workspace / "study.json") != receipt["study_sha256"]:
        raise ValueError("Local configuration changed after preparation")
    for name, expected in receipt["control_sha256"].items():
        if digest(workspace / name) != expected:
            raise ValueError(f"Prepared launch file changed: {name}")
    release = Path(receipt["release"])
    if digest(release / "PACKAGE_SHA256.json") != receipt["package_sha256"]:
        raise ValueError("Prepared release manifest changed")
    for name, expected in json.loads((release / "PACKAGE_SHA256.json").read_text()).items():
        if digest(release / name) != expected:
            raise ValueError(f"Prepared release file changed: {name}")
    values = json.loads((workspace / "study.json").read_text())
    output = Path(values["output_dir"])
    if output.exists():
        raise FileExistsError("Refusing to restart over existing local run output")
    python, release = Path(receipt["python"]), Path(receipt["release"])
    environment = local_environment(runtime, workspace, release)
    sys.path.insert(0, str(workspace))
    from launch_full_run import select_devices

    hardware = workspace / ("hardware-" + os.environ["SLURM_JOB_ID"])
    hardware.mkdir()
    command(
        [python, workspace / "probe_allocated_gpus.py", "--output", hardware / "probes.json", "--timeout", "300"],
        env=environment,
        timeout=360,
    )
    probes = json.loads((hardware / "probes.json").read_text())
    devices = select_devices(probes, 8, values["trainer_gpus"] + values["inference_gpus"])
    if any(
        "A100" not in row["name"] or row["bytes"] < 79_000_000_000
        for row in probes["results"]
        if row["device"] in devices
    ):
        raise ValueError("Local run requires A100 80GB GPUs")
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    command(
        [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={len(devices)}",
            release / "scripts/gpu_health.py",
            "--expected-gpus",
            str(len(devices)),
            "--receipt",
            hardware / "collective.json",
        ],
        env=environment,
        timeout=900,
        cwd=release,
    )
    result = json.loads((hardware / "collective.json").read_text())
    if result["world_size"] != len(devices) or len(result["devices"]) != len(devices):
        raise ValueError("Collective verification did not cover the prepared topology")
    stop = workspace / "backup.stop"
    status = workspace / "backup-status.json"
    backup_log = (workspace / "backup.log").open("a")
    training_log = (workspace / "training.log").open("x")
    backup = subprocess.Popen(
        [
            str(python),
            str(workspace / "local_backup.py"),
            "--source",
            str(output),
            "--destination",
            spec["backup"],
            "--metrics",
            spec["metrics"],
            "--stop-file",
            str(stop),
            "--status",
            str(status),
            "--lock",
            str(runtime.parent / "backup-transfer.lock"),
        ],
        env=environment,
        stdout=backup_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    training = subprocess.Popen(
        [str(python), "-m", "deepseek_study.cli", "run", str(workspace / "study.json")],
        env=environment,
        cwd=release,
        stdout=training_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    interrupted = False

    def interrupt(*_):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    atomic_json(
        workspace / "supervisor.json",
        {
            "job_id": os.environ["SLURM_JOB_ID"],
            "node": receipt["node"],
            "training_pid": training.pid,
            "backup_pid": backup.pid,
            "local_output": str(output),
            "shared_backup": spec["backup"],
            "devices": devices,
            "threads_per_worker": 1,
        },
    )
    started = time.monotonic()
    inference_ready = False
    failure = None
    try:
        while training.poll() is None:
            if interrupted:
                raise RuntimeError("Local supervisor received termination")
            if backup.poll() is not None:
                raise RuntimeError("Background backup process exited; refusing unprotected training")
            if shutil.disk_usage(workspace).free < 50 * 1024**3:
                raise RuntimeError("Local disk reserve fell below 50 GiB")
            if output.is_dir() and not (output / "storage/ready.json").exists():
                atomic_json(output / "storage/ready.json", receipt)
                atomic_json(
                    output / "storage/supervisor.json", json.loads((workspace / "supervisor.json").read_text())
                )
            if output.is_dir() and status.is_file():
                atomic_json(output / "storage/backup-status.json", json.loads(status.read_text()))
            if not inference_ready:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{values['inference_port']}/health", timeout=2
                    ) as response:
                        inference_ready = response.status == 200
                except OSError:
                    pass
                if time.monotonic() - started > 1800 and not inference_ready:
                    raise RuntimeError("Inference was not ready within the 30-minute startup deadline")
            time.sleep(5)
        if training.returncode:
            raise RuntimeError(f"Training exited with code {training.returncode}")
    except BaseException as error:
        failure = repr(error)
        if training.poll() is None:
            os.killpg(training.pid, signal.SIGTERM)
            try:
                training.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(training.pid, signal.SIGKILL)
                training.wait(timeout=30)
    finally:
        stop.touch()
        try:
            backup.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            failure = failure or "Final backup exceeded one hour; local files retained"
            os.killpg(backup.pid, signal.SIGTERM)
        atomic_json(
            workspace / "supervisor-complete.json",
            {
                "training_exit": training.poll(),
                "backup_exit": backup.poll(),
                "error": failure,
                "finished_at": time.time(),
                "local_files_retained": True,
            },
        )
    if failure or backup.poll() != 0:
        raise RuntimeError(failure or "Final backup failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "run"))
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    control = args.control.resolve()
    specification = json.loads((control / "storage-spec.json").read_text())
    if args.mode == "stage":
        for name, expected in json.loads((control / "CONTROL_SHA256.json").read_text()).items():
            if digest(control / name) != expected:
                raise ValueError(f"Staging control changed: {name}")
        stage(specification, control)
    else:
        run(specification)


if __name__ == "__main__":
    main()
