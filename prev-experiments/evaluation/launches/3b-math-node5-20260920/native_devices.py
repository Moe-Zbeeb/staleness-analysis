import ctypes
import hashlib
import os
import re
import socket
import uuid
from pathlib import Path

GPU_COUNT = 3


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


def visible_inventory():
    require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    require(os.environ.get("CUDA_VISIBLE_DEVICES"), "Preserve Slurm's CUDA_VISIBLE_DEVICES")
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
    require(count.value == GPU_COUNT, f"Slurm must expose exactly three CUDA devices, got {count.value}")
    result = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        call("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.byref(device), ordinal)
        identifier = (ctypes.c_ubyte * 16)()
        call("cuDeviceGetUuid_v2", [ctypes.c_void_p, ctypes.c_int], ctypes.byref(identifier), device.value)
        bus = ctypes.create_string_buffer(64)
        call("cuDeviceGetPCIBusId", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], bus, len(bus), device.value)
        result.append({"local_ordinal": ordinal, "uuid": str(uuid.UUID(bytes=bytes(identifier))), "pci_bus_id": bus.value.decode()})
    require(len({entry["uuid"] for entry in result}) == GPU_COUNT, "CUDA inventory has duplicate UUIDs")
    return result


def bind_visible_devices(cvd, allocated, visible, physical):
    selectors = parse_ids(cvd)
    require(len(selectors) == GPU_COUNT, "Expected three original Slurm visible selectors")
    require(selectors == parse_ids(allocated), "Unexpected Slurm selector remapping on the verified non-cgroup node")
    require([entry["local_ordinal"] for entry in visible] == list(range(GPU_COUNT)), "Incomplete visible CUDA inventory")
    require(len({entry["uuid"] for entry in visible}) == GPU_COUNT, "Duplicate visible physical GPUs")
    by_uuid = {entry["uuid"]: entry for entry in physical.values()}
    require(len(by_uuid) == len(physical), "Physical GPU inventory has duplicate UUIDs")
    result = []
    for selector, entry in zip(selectors, visible, strict=True):
        require(entry["uuid"] in by_uuid, "A Slurm-visible GPU is missing from the physical inventory")
        actual = by_uuid[entry["uuid"]]
        require(entry["pci_bus_id"].lower() == actual["pci_bus_id"].lower(), "CUDA and physical PCI identity differ")
        result.append({"slurm_visible_selector": selector, "device_file": f"/dev/nvidia{actual['minor']}", **actual, **entry})
    return result


def inspect_assignment():
    require(socket.gethostname().split(".")[0] == "deep-chungus-5", "Native-selector recovery is validated for deep-chungus-5 only")
    require(os.environ.get("SLURM_JOB_NUM_NODES", "1") == "1", "Expected one node")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    require(cvd and cvd == os.environ.get("RECOVERY_SLURM_CUDA_VISIBLE_DEVICES"), "Original Slurm CUDA_VISIBLE_DEVICES changed")
    devices = bind_visible_devices(cvd, os.environ["SLURM_JOB_GPUS"], visible_inventory(), proc_inventory())
    return cvd, devices
