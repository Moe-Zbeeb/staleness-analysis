import argparse
import ctypes
import csv
import io
import datetime
import hashlib
import json
import os
import re
import shlex
import socket
import stat
import subprocess
import sys
import uuid
from pathlib import Path


GPU_COUNT = 6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize_uuid(value):
    return str(uuid.UUID(value.removeprefix("GPU-")))


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
        fields = dict(token.split("=", 1) for token in tokens)
        require(not any(key.lower() == "include" for key in fields), "GRES Include directives require explicit resolution")
        if "NodeName" in fields:
            require(fields["NodeName"] != "DEFAULT", "GRES NodeName=DEFAULT is unsupported")
            if node not in expand_expression(fields["NodeName"].rstrip(",")):
                continue
        if fields.get("Name") != "gpu":
            continue
        require(fields.get("AutoDetect") == "off", "Only reviewed explicit GRES File mappings with AutoDetect=off are supported")
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
    for path in root.glob("*/information"):
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
    require(len(allocated) == GPU_COUNT and len(set(allocated)) == GPU_COUNT, "Expected exactly six distinct Slurm GPU IDs")
    result = []
    for gres_id in allocated:
        require(0 <= gres_id < len(configured_files), f"Slurm GPU ID {gres_id} is outside configured GRES devices")
        device_file = configured_files[gres_id]
        minor = int(device_file.removeprefix("/dev/nvidia"))
        require(minor in inventory, f"Allocated GPU {device_file} has no NVIDIA identity")
        result.append({"slurm_gpu_id": gres_id, "device_file": device_file, **inventory[minor]})
    return result


def verify_device_files(devices):
    for entry in devices:
        metadata = Path(entry["device_file"]).stat()
        require(stat.S_ISCHR(metadata.st_mode), f"Allocated GPU is not a character device: {entry['device_file']}")
        require(os.minor(metadata.st_rdev) == entry["minor"], f"GPU device-file minor differs from NVIDIA mapping: {entry['device_file']}")


def verify_idle_devices(devices):
    selected = ['GPU-' + entry['uuid'] for entry in devices]
    selector = '--id=' + ','.join(selected)
    command = ['nvidia-smi', selector, '--query-gpu=uuid,name,memory.total,memory.free', '--format=csv,noheader,nounits']
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    hardware = {}
    for row in csv.reader(io.StringIO(result.stdout)):
        require(len(row) == 4, 'Unexpected allocated GPU memory query schema')
        device_uuid, name, total, free = [value.strip() for value in row]
        require(device_uuid in selected and device_uuid not in hardware, 'GPU memory query returned an unallocated or duplicate UUID')
        total, free = int(total), int(free)
        require(any(kind in name for kind in ('A100', 'H100')) and total / 1024 >= 75, f'Allocated GPU {device_uuid} is not the required A100/H100 80GB class')
        require(0 <= free <= total and free >= 0.85 * total, f'Allocated GPU {device_uuid} has insufficient free VRAM')
        hardware[device_uuid] = {'uuid': device_uuid, 'name': name, 'total_memory_mib': total, 'free_memory_mib': free, 'minimum_free_fraction': 0.85}
    require(set(hardware) == set(selected), 'GPU memory query omitted an allocated UUID')
    command = ['nvidia-smi', selector, '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits']
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    occupied = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if not row or row == ['No running processes found']:
            continue
        require(len(row) == 3, 'Unexpected allocated GPU process query schema')
        device_uuid, pid, memory = [value.strip() for value in row]
        require(device_uuid in selected and pid.isdigit(), 'GPU process query returned an invalid or unallocated device')
        occupied.append({'uuid': device_uuid, 'pid': int(pid), 'used_memory_mib': memory})
    require(not occupied, f'Allocated GPUs already have compute processes; no process was signaled: {occupied}')
    return {'status': 'passed', 'allocated_gpu_uuids': selected, 'hardware': [hardware[value] for value in selected], 'existing_compute_processes': occupied}


def cuda_driver_inventory():
    require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    require("CUDA_VISIBLE_DEVICES" not in os.environ, "CUDA inventory subprocess must not have a CVD filter")
    driver = ctypes.CDLL("libcuda.so.1")

    def call(name, argtypes, *arguments):
        function = getattr(driver, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        status = function(*arguments)
        require(status == 0, f"CUDA driver query {name} failed with error {status}")

    call("cuInit", [ctypes.c_uint], 0)
    count = ctypes.c_int()
    call("cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)], ctypes.byref(count))
    require(GPU_COUNT <= count.value <= 256, f"Unexpected CUDA device count: {count.value}")
    result = []
    uuid_function = "cuDeviceGetUuid_v2" if hasattr(driver, "cuDeviceGetUuid_v2") else "cuDeviceGetUuid"
    for ordinal in range(count.value):
        device = ctypes.c_int()
        call("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.byref(device), ordinal)
        identifier = (ctypes.c_ubyte * 16)()
        call(uuid_function, [ctypes.c_void_p, ctypes.c_int], ctypes.byref(identifier), device.value)
        bus = ctypes.create_string_buffer(64)
        call("cuDeviceGetPCIBusId", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], bus, len(bus), device.value)
        result.append({"cuda_ordinal": ordinal, "uuid": str(uuid.UUID(bytes=bytes(identifier))), "pci_bus_id": bus.value.decode()})
    require(len({entry["uuid"] for entry in result}) == count.value, "CUDA inventory has duplicate UUIDs")
    return result


def numeric_selectors(expected, inventory):
    by_uuid = {entry["uuid"]: entry["cuda_ordinal"] for entry in inventory}
    require(len(by_uuid) == len(inventory), "CUDA inventory UUIDs are not unique")
    require(len(set(by_uuid.values())) == len(inventory), "CUDA inventory ordinals are not unique")
    require(all(entry["uuid"] in by_uuid for entry in expected), "An allocated GPU is absent from CUDA driver enumeration")
    selected = [by_uuid[entry["uuid"]] for entry in expected]
    require(len(selected) == GPU_COUNT and len(set(selected)) == GPU_COUNT, "CUDA selection must contain six unique GPUs")
    return selected


def prepare(args):
    require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "Export CUDA_DEVICE_ORDER=PCI_BUS_ID before preparing GPU mapping")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    require(job_id.isdigit(), "GPU mapping requires an active Slurm allocation")
    node = socket.gethostname().split(".")[0]
    require(os.environ.get("SLURM_JOB_NUM_NODES", "1") == "1", "GPU mapping requires a single-node allocation")
    contents = args.gres_config.read_text()
    digest = hashlib.sha256(contents.encode()).hexdigest()
    require(digest == args.expected_gres_sha256, "Live GRES configuration differs from reviewed source")
    configured, records = gres_files(contents, node)
    allocated = parse_ids(os.environ["SLURM_JOB_GPUS"])
    expected = expected_devices(allocated, configured, proc_inventory())
    verify_device_files(expected)
    idle_proof = verify_idle_devices(expected)
    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    query = subprocess.run([sys.executable, str(Path(__file__).resolve()), "driver-inventory"], env=environment, check=True, capture_output=True, text=True, timeout=45)
    inventory = json.loads(query.stdout)
    selected = numeric_selectors(expected, inventory)
    visible = ",".join(map(str, selected))
    result = {
        "schema": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "job_id": job_id,
        "restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
        "node": node,
        "cuda_device_order": "PCI_BUS_ID",
        "original_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_gpus": os.environ["SLURM_JOB_GPUS"],
        "slurm_gpu_ids": allocated,
        "gres_config_path": str(args.gres_config.resolve()),
        "gres_config_sha256": digest,
        "gres_records": records,
        "configured_gpu_files": configured,
        "allocated_devices": expected,
        "prelaunch_gpu_gate": idle_proof,
        "cuda_driver_inventory": inventory,
        "cuda_visible_devices": visible,
        "prime_visible_devices": ",".join(map(str, selected[4:] + selected[:4])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(visible, flush=True)


def read_mapping(path):
    data = path.read_bytes()
    record = json.loads(data)
    require(record["schema"] == 1, "Unsupported GPU mapping schema")
    require(record["job_id"] == os.environ.get("SLURM_JOB_ID"), "GPU mapping belongs to another job")
    require(record["restart_count"] == os.environ.get("SLURM_RESTART_COUNT", "0"), "GPU mapping belongs to another restart attempt")
    require(record["node"] == socket.gethostname().split(".")[0], "GPU mapping belongs to another node")
    require(record["cuda_device_order"] == os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA device ordering changed")
    require(record["cuda_visible_devices"] == os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA device selection differs from mapping")
    require(record["slurm_gpu_ids"] == parse_ids(os.environ["SLURM_JOB_GPUS"]), "Slurm allocation changed")
    config = Path(record["gres_config_path"]).read_text()
    require(hashlib.sha256(config.encode()).hexdigest() == record["gres_config_sha256"], "GRES configuration changed after mapping")
    configured, records = gres_files(config, record["node"])
    require(configured == record["configured_gpu_files"] and records == record["gres_records"], "GRES mapping evidence changed")
    expected = expected_devices(record["slurm_gpu_ids"], configured, proc_inventory())
    verify_device_files(expected)
    require(expected == record["allocated_devices"], "Allocated physical GPU identity changed")
    require(",".join(map(str, numeric_selectors(expected, record["cuda_driver_inventory"]))) == record["cuda_visible_devices"], "CUDA ordinal mapping is inconsistent")
    return record, hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("driver-inventory")
    command = commands.add_parser("prepare")
    command.add_argument("--gres-config", type=Path, required=True)
    command.add_argument("--expected-gres-sha256", required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "driver-inventory":
        print(json.dumps(cuda_driver_inventory()), flush=True)
    else:
        prepare(args)


if __name__ == "__main__":
    main()
