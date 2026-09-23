import datetime
import hashlib
import json
import os
import socket
from pathlib import Path

import native_devices


def allocation(args, frozen):
    job_id = os.environ.get("SLURM_JOB_ID", "")
    native_devices.require(job_id.isdigit(), "GPU selection requires a Slurm allocation")
    native_devices.require(os.environ.get("SLURM_GPUS_ON_NODE") == "2", "This evaluation requires two assigned GPUs")
    config_path = Path(args.gres_config)
    contents = config_path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    native_devices.require(digest == args.expected_gres_sha256, "Reviewed GRES configuration changed")
    cvd, devices = native_devices.inspect_assignment()
    devices = [{**device, "uuid": "GPU-" + device["uuid"]} for device in devices]
    node = socket.gethostname().split(".")[0]
    configured, records = frozen.gres_files(contents.decode(), node)
    frozen.verify_device_files(devices)
    memory = {key: int(value.split()[0]) * 1024 for key, value in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()) if key in ["MemTotal", "MemAvailable", "MemFree", "Cached"]}
    minimum = 24 * 1024**3
    memory_record = {"job_id": job_id, "at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "memory_bytes": memory, "required_available_bytes": minimum, "passed": memory["MemAvailable"] >= minimum}
    directory = args.results_root / ".native-recovery" / f"{job_id}-{os.environ.get('SLURM_RESTART_COUNT', '0')}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "host-memory.json").write_text(json.dumps(memory_record, indent=2) + "\n")
    native_devices.require(memory_record["passed"], "Insufficient available host RAM for the two-worker evaluation")
    hardware, cohort = frozen.query_hardware(devices, node)
    return {
        "schema_version": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "job_id": job_id,
        "restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
        "host": socket.gethostname(),
        "gpu_count": len(devices),
        "worker_gpu_uuids": [entry["uuid"] for entry in devices],
        "original_cuda_visible_devices": cvd,
        "cuda_device_order": os.environ["CUDA_DEVICE_ORDER"],
        "slurm_job_gpus": os.environ["SLURM_JOB_GPUS"],
        "slurm_gpu_ids": native_devices.parse_ids(os.environ["SLURM_JOB_GPUS"]),
        "gres_config_path": str(config_path.resolve()),
        "gres_config_sha256": digest,
        "gres_records": records,
        "configured_gpu_files": configured,
        "allocated_devices": devices,
        "inventory": hardware,
        "hardware_cohort": cohort,
        "host_memory": memory_record,
        "assignment_basis": "Original Slurm CUDA_VISIBLE_DEVICES with PCI_BUS_ID ordering; identities resolved only within that visible set",
    }
