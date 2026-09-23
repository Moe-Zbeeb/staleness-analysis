import argparse
import csv
import datetime
import hashlib
import io
import json
import os
import re
import shlex
import socket
import stat
import subprocess
import uuid
from pathlib import Path


GPU_COUNT = 2
ALLOWED_NODES = {"deep-chungus-1", "deep-chungus-11"}
HARDWARE_CLASSES = {"deep-chungus-1": ("a100-40gb", 37, 45), "deep-chungus-11": ("a100-80gb", 75, 85)}
MINIMUM_FREE_FRACTION = 0.85


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize_uuid(value):
    require(value.startswith("GPU-"), "Expected a full NVIDIA GPU UUID")
    return "GPU-" + str(uuid.UUID(value[4:]))


def parse_ids(value):
    values = []
    for part in value.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:-(\d+))?\s*", part)
        require(match is not None, f"Unsupported GPU ID expression: {value!r}")
        first = int(match.group(1))
        last = int(match.group(2) or first)
        require(first <= last < 256, f"Invalid GPU ID range: {part!r}")
        values.extend(range(first, last + 1))
    require(len(values) == len(set(values)), f"Duplicate GPU IDs: {value!r}")
    return values


def expand_expression(value):
    parts = []
    depth = 0
    start = 0
    for index, char in enumerate(value):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(value[start:index])
            start = index + 1
        require(depth in (0, 1), f"Unsupported hostlist expression: {value!r}")
    require(depth == 0, f"Unclosed hostlist expression: {value!r}")
    parts.append(value[start:])
    result = []
    for part in parts:
        match = re.fullmatch(r"([^\[\]]*)\[([^\[\]]+)\]([^\[\]]*)", part)
        if match is None:
            require(part and "[" not in part and "]" not in part, f"Invalid expression: {part!r}")
            result.append(part)
        else:
            require(not re.search(r"\b0\d", match.group(2)), "Zero-padded hostlist ranges are unsupported")
            result.extend(f"{match.group(1)}{number}{match.group(3)}" for number in parse_ids(match.group(2)))
    require(len(result) <= 256 and len(result) == len(set(result)), "Hostlist must contain unique bounded entries")
    return result


def gres_files(contents, node):
    files = []
    records = []
    for line_number, line in enumerate(contents.splitlines(), 1):
        tokens = shlex.split(line, comments=True)
        if not tokens:
            continue
        require(all("=" in token for token in tokens), f"Unsupported GRES directive at line {line_number}")
        pairs = [token.split("=", 1) for token in tokens]
        fields = dict(pairs)
        require(len(fields) == len(pairs), f"Duplicate GRES fields at line {line_number}")
        require(not any(key.lower() == "include" for key in fields), "GRES Include directives require explicit resolution")
        if "NodeName" in fields:
            require(fields["NodeName"] != "DEFAULT", "GRES NodeName=DEFAULT is unsupported")
            if node not in expand_expression(fields["NodeName"].rstrip(",")):
                continue
        if fields.get("Name") != "gpu":
            continue
        require(fields.get("AutoDetect") == "off", "Only explicit GRES File mappings with AutoDetect=off are supported")
        require(fields.get("Type") == "a100", "The authorized nodes require A100 GPU GRES")
        require("File" in fields and "MultipleFiles" not in fields, "GPU GRES must have explicit full-device File entries")
        expanded = expand_expression(fields["File"])
        require(all(re.fullmatch(r"/dev/nvidia\d+", path) for path in expanded), "Only full NVIDIA GPU device files are supported")
        if "Count" in fields:
            require(fields["Count"].isdigit() and int(fields["Count"]) == len(expanded), "GPU GRES Count differs from explicit device files")
        records.append({"line": line_number, "fields": fields, "files": expanded})
        files.extend(expanded)
    require(files and len(files) == len(set(files)), "No unique explicit GPU GRES File mapping for this node")
    require(files == sorted(files, key=lambda path: int(path.removeprefix("/dev/nvidia"))), "GRES device File ordering must be increasing numeric order")
    return files, records


def proc_inventory(root=Path("/proc/driver/nvidia/gpus")):
    devices = {}
    for path in sorted(root.glob("*/information")):
        contents = path.read_text()
        minor = re.search(r"Device Minor:\s*(\d+)", contents)
        device_uuid = re.search(r"GPU UUID:\s*(\S+)", contents)
        require(minor and device_uuid, f"Missing GPU identity fields in {path}")
        number = int(minor.group(1))
        require(number not in devices, f"Duplicate NVIDIA minor number: {number}")
        devices[number] = {
            "minor": number,
            "uuid": normalize_uuid(device_uuid.group(1)),
            "pci_bus_id": path.parent.name,
            "information_path": str(path),
            "information_sha256": hashlib.sha256(contents.encode()).hexdigest(),
        }
    require(devices, "NVIDIA /proc GPU inventory is empty")
    require(len({entry["uuid"] for entry in devices.values()}) == len(devices), "Duplicate NVIDIA GPU UUIDs")
    return devices


def expected_devices(allocated, configured_files, inventory):
    require(len(allocated) == GPU_COUNT and len(set(allocated)) == GPU_COUNT, "Expected exactly two distinct Slurm GPU IDs")
    result = []
    for gres_id in allocated:
        require(0 <= gres_id < len(configured_files), f"Slurm GPU ID {gres_id} is outside configured GRES devices")
        device_file = configured_files[gres_id]
        minor = int(device_file.removeprefix("/dev/nvidia"))
        require(minor in inventory, f"Allocated GPU {device_file} has no NVIDIA identity")
        entry = inventory[minor]
        require(entry["minor"] == minor, "NVIDIA inventory minor is inconsistent")
        require(normalize_uuid(entry["uuid"]) == entry["uuid"], "NVIDIA inventory UUID is not canonical")
        result.append({"slurm_gpu_id": gres_id, "device_file": device_file, **entry})
    require(len({entry["uuid"] for entry in result}) == GPU_COUNT, "Allocated NVIDIA GPU UUIDs are not unique")
    return result


def verify_device_files(devices):
    for entry in devices:
        metadata = Path(entry["device_file"]).stat()
        require(stat.S_ISCHR(metadata.st_mode), f"Allocated GPU is not a character device: {entry['device_file']}")
        require(os.minor(metadata.st_rdev) == entry["minor"], f"GPU device-file minor differs from NVIDIA mapping: {entry['device_file']}")


def query_hardware(devices, node):
    selected = [entry["uuid"] for entry in devices]
    selector = "--id=" + ",".join(selected)
    command = ["nvidia-smi", selector, "--query-gpu=uuid,name,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, check=True, text=True, capture_output=True, timeout=30)
    hardware = {}
    cohort, minimum, maximum = HARDWARE_CLASSES[node]
    for row in csv.reader(io.StringIO(completed.stdout)):
        if not row:
            continue
        require(len(row) == 5, "Unexpected NVIDIA hardware query schema")
        device_uuid, name, total, free, used = (part.strip() for part in row)
        device_uuid = normalize_uuid(device_uuid)
        require(device_uuid in selected and device_uuid not in hardware, "NVIDIA query returned an unallocated or duplicate GPU")
        total, free, used = int(total), int(free), int(used)
        require("A100" in name and minimum <= total / 1024 < maximum, f"Allocated GPU is outside the expected {cohort} hardware cohort")
        require(0 <= free <= total and 0 <= used <= total, "Invalid NVIDIA memory inventory")
        require(free >= total * MINIMUM_FREE_FRACTION, f"Allocated GPU {device_uuid} has insufficient free VRAM for the frozen 85% vLLM reservation")
        hardware[device_uuid] = {"uuid": device_uuid, "name": name, "total_memory_mib": total, "free_memory_mib": free, "used_memory_mib": used, "hardware_cohort": cohort, "minimum_free_fraction": MINIMUM_FREE_FRACTION, "compute_processes": []}
    require(set(hardware) == set(selected), "NVIDIA hardware query omitted an allocated GPU")
    command = ["nvidia-smi", selector, "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, check=True, text=True, capture_output=True, timeout=30)
    for row in csv.reader(io.StringIO(completed.stdout)):
        if not row or row == ["No running processes found"]:
            continue
        require(len(row) == 4, "Unexpected NVIDIA compute-process query schema")
        device_uuid, pid, name, memory = (part.strip() for part in row)
        device_uuid = normalize_uuid(device_uuid)
        require(device_uuid in hardware, "NVIDIA process query returned an unallocated GPU")
        require(pid.isdigit(), "NVIDIA compute-process PID is invalid")
        hardware[device_uuid]["compute_processes"].append({"pid": int(pid), "name": name, "used_memory_mib": int(memory) if memory.isdigit() else None, "reported_used_memory": memory})
    return [hardware[device_uuid] for device_uuid in selected], cohort


def allocation(args):
    job_id = os.environ.get("SLURM_JOB_ID", "")
    require(job_id.isdigit(), "GPU mapping requires an active Slurm allocation")
    require(os.environ.get("SLURM_JOB_NUM_NODES", "") == "1", "GPU mapping requires a single-node allocation")
    require(os.environ.get("SLURM_GPUS_ON_NODE", "") == str(GPU_COUNT), "GPU mapping requires exactly two allocated GPUs on the node")
    host = socket.gethostname()
    node = host.split(".")[0]
    require(node in ALLOWED_NODES, f"Node is outside this authorized evaluation: {node}")
    config_path = Path(args.gres_config)
    contents = config_path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    require(re.fullmatch(r"[a-f0-9]{64}", args.expected_gres_sha256) is not None, "A reviewed GRES SHA256 is required")
    require(digest == args.expected_gres_sha256, "Live GRES configuration differs from reviewed source")
    configured, records = gres_files(contents.decode(), node)
    allocated = parse_ids(os.environ.get("SLURM_JOB_GPUS", ""))
    devices = expected_devices(allocated, configured, proc_inventory())
    verify_device_files(devices)
    hardware, cohort = query_hardware(devices, node)
    return {
        "schema_version": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "job_id": job_id,
        "restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
        "host": host,
        "gpu_count": len(devices),
        "worker_gpu_uuids": [entry["uuid"] for entry in devices],
        "original_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_gpus": os.environ["SLURM_JOB_GPUS"],
        "slurm_gpu_ids": allocated,
        "gres_config_path": str(config_path.resolve()),
        "gres_config_sha256": digest,
        "gres_records": records,
        "configured_gpu_files": configured,
        "allocated_devices": devices,
        "inventory": hardware,
        "hardware_cohort": cohort,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gres-config", type=Path, required=True)
    parser.add_argument("--expected-gres-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(allocation(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
