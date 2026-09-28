import argparse
import json
import os
from pathlib import Path

from local_backup import atomic_json, digest
from node_local_run import command, local_environment, stage


CONTROL_FILES = (
    "launch_historical_worker.py",
    "node_local_run.py",
    "local_backup.py",
    "launch_full_run.py",
    "probe_allocated_gpus.py",
)


def validate_allocation(environment, count):
    devices = environment.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if (
        not environment.get("SLURM_JOB_ID")
        or not environment.get("SLURMD_NODENAME")
        or int(environment.get("SLURM_RESTART_COUNT", "0"))
        or type(count) is not int
        or count < 1
        or len(devices) != count
        or len(set(devices)) != count
        or not all(devices)
    ):
        raise ValueError("Historical worker requires a fresh Slurm allocation matching its inference GPU count")
    return devices


def validate_preparation(spec, control):
    control = Path(control)
    required = ("baseline_study", "release", "shared_prime", "workspace", "runtime")
    if any(not isinstance(spec.get(key), str) or not Path(spec[key]).is_absolute() for key in required):
        raise ValueError("Historical worker staging requires explicit absolute source and local paths")
    if (
        not isinstance(spec.get("run_name"), str)
        or not spec["run_name"]
        or spec["run_name"] in {".", ".."}
        or Path(spec["run_name"]).name != spec["run_name"]
    ):
        raise ValueError("Historical worker output name must be one path component")
    release = Path(spec["release"])
    package = json.loads((release / "PACKAGE_SHA256.json").read_text())
    for name in CONTROL_FILES:
        expected = package.get("scripts/" + name)
        source = control / name
        if expected is None or source.is_symlink() or not source.is_file() or digest(source) != expected:
            raise ValueError(f"Historical worker control differs from the frozen release: {name}")
    baseline = json.loads(Path(spec["baseline_study"]).read_text())
    if (
        type(baseline.get("inference_gpus")) is not int
        or baseline["inference_gpus"] < 1
        or baseline.get("inference_tensor_parallel") != 1
    ):
        raise ValueError("Historical worker requires an explicit inference pool with tensor parallel one")
    history = baseline.get("historical_rollouts")
    if not isinstance(history, str) or not Path(history).is_absolute() or baseline.get("lag", 0) < 1:
        raise ValueError("Historical worker requires the learner's shared queue and positive exact lag")
    manifest = json.loads(Path(baseline["data_manifest"]).read_text())
    dataset_hash = digest(Path(baseline["dataset_path"]))
    if manifest.get("contract", {}).get("source_sha256") != dataset_hash:
        raise ValueError("Historical worker dataset and prepared data manifest differ")
    return baseline


def worker_environment(runtime, workspace, release):
    environment = local_environment(runtime, workspace, release)
    for key in list(environment):
        if key.startswith(("NCCL_", "TORCHELASTIC_")) or key in {
            "DEEPSEEK_STUDY_REMOTE_INFERENCE",
            "GLOO_SOCKET_IFNAME",
            "VLLM_HOST_IP",
            "MASTER_ADDR",
            "MASTER_PORT",
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
        }:
            environment.pop(key)
    return environment


def validate_probes(probes, allocated, environment, min_gpu_bytes=39_000_000_000):
    if (
        probes.get("allocated_devices") != allocated
        or probes.get("healthy_devices") != allocated
        or str(probes.get("job_id")) != str(environment["SLURM_JOB_ID"])
        or probes.get("node") != environment["SLURMD_NODENAME"]
    ):
        raise ValueError("Historical GPU probes do not match the current allocation")
    records = probes.get("results")
    if not isinstance(records, list) or len(records) != len(allocated):
        raise ValueError("Historical worker GPU probes are incomplete")
    seen_devices, seen_uuids = set(), set()
    for record in records:
        identity = record.get("uuid")
        total, free = record.get("initial_total_bytes", 0), record.get("initial_free_bytes", 0)
        if (
            not record.get("healthy")
            or record.get("device") not in allocated
            or record.get("device") in seen_devices
            or not isinstance(identity, str)
            or identity in {"", "unavailable"}
            or identity in seen_uuids
            or "A100" not in record.get("name", "")
            or record.get("bytes", 0) < min_gpu_bytes
            or total < min_gpu_bytes
            or free < 0.9 * total
            or record.get("bf16_backward") is not True
        ):
            raise ValueError("Historical worker requires healthy, unoccupied A100 GPUs meeting the memory minimum")
        seen_devices.add(record["device"])
        seen_uuids.add(identity)
    return allocated


def verify_staged(spec, receipt):
    workspace = Path(spec["workspace"])
    if receipt.get("node") != os.environ["SLURMD_NODENAME"]:
        raise ValueError("Historical worker staging belongs to another node")
    if digest(workspace / "study.json") != receipt["study_sha256"]:
        raise ValueError("Historical worker study changed after staging")
    release = Path(receipt["release"])
    if digest(release / "PACKAGE_SHA256.json") != receipt["package_sha256"]:
        raise ValueError("Historical worker package manifest changed after staging")
    for name, expected in json.loads((release / "PACKAGE_SHA256.json").read_text()).items():
        if digest(release / name) != expected:
            raise ValueError(f"Historical worker source changed after staging: {name}")
    for name, expected in receipt["control_sha256"].items():
        if digest(workspace / name) != expected:
            raise ValueError(f"Historical worker staged control changed: {name}")
    study = json.loads((workspace / "study.json").read_text())
    if Path(study["output_dir"]).exists():
        raise FileExistsError("Refusing to restart over existing historical worker output")
    return release


def launch(control, learner_job):
    control = Path(control).resolve()
    if type(learner_job) is not int or learner_job < 1:
        raise ValueError("Historical worker needs an explicit learner allocation ID")
    spec = json.loads((control / "storage-spec.json").read_text())
    baseline = validate_preparation(spec, control)
    allocated = validate_allocation(os.environ, baseline["inference_gpus"])
    if str(learner_job) == str(os.environ["SLURM_JOB_ID"]):
        raise ValueError("Historical worker and learner require separate allocations")
    receipt = stage(spec, control)
    release = verify_staged(spec, receipt)
    workspace, runtime = Path(spec["workspace"]), Path(spec["runtime"])
    python = Path(receipt["python"])
    environment = worker_environment(runtime, workspace, release)
    hardware = workspace / ("worker-hardware-" + os.environ["SLURM_JOB_ID"])
    hardware.mkdir(exist_ok=False)
    command(
        [python, workspace / "probe_allocated_gpus.py", "--output", hardware / "probes.json", "--timeout", "300"],
        env=environment,
        timeout=360,
        cwd=release,
    )
    probes = json.loads((hardware / "probes.json").read_text())
    minimum_gpu_bytes = spec.get("minimum_gpu_bytes", 39_000_000_000)
    if type(minimum_gpu_bytes) is not int or minimum_gpu_bytes < 39_000_000_000:
        raise ValueError("Historical worker GPU memory minimum must be at least 39 billion bytes")
    devices = validate_probes(probes, allocated, os.environ, minimum_gpu_bytes)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    invocation = [
        str(python),
        "-m",
        "deepseek_study.runtime.historical_worker",
        "launch",
        str(workspace / "study.json"),
        "--learner-job",
        str(learner_job),
    ]
    record = {
        "job_id": os.environ["SLURM_JOB_ID"],
        "learner_job_id": learner_job,
        "node": os.environ["SLURMD_NODENAME"],
        "inference_devices": devices,
        "trainer_devices": [],
        "cross_node_nccl": False,
        "historical_rollouts": baseline["historical_rollouts"],
        "local_output": json.loads((workspace / "study.json").read_text())["output_dir"],
        "study_sha256": receipt["study_sha256"],
        "package_sha256": receipt["package_sha256"],
        "launcher_sha256": digest(control / "launch_historical_worker.py"),
        "probes": probes,
        "command": invocation,
    }
    atomic_json(workspace / "historical-worker-launch.json", record)
    atomic_json(control / ("historical-worker-launch-" + os.environ["SLURM_JOB_ID"] + ".json"), record)
    os.chdir(release)
    os.execve(python, invocation, environment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--learner-job", type=int, required=True)
    args = parser.parse_args()
    launch(args.control, args.learner_job)


if __name__ == "__main__":
    main()
