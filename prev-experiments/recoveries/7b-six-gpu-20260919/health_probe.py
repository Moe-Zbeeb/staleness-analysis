import json
import os
import re
import socket
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def allocated_minors() -> set[int]:
    value = os.environ["SLURM_JOB_GPUS"]
    allocated = []
    for part in value.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:-(\d+))?\s*", part)
        if match is None:
            raise ValueError(f"Unsupported SLURM_JOB_GPUS value: {value!r}")
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if last < first:
            raise ValueError(f"Invalid GPU range: {part!r}")
        allocated.extend(range(first, last + 1))
    if len(allocated) != 6 or len(set(allocated)) != 6:
        raise ValueError(f"Expected six distinct allocated GPU minors: {value!r}")
    return set(allocated)


def allocated_uuids() -> set[str]:
    allocated = allocated_minors()
    mapped = {}
    for path in Path("/proc/driver/nvidia/gpus").glob("*/information"):
        contents = path.read_text()
        minor_match = re.search(r"Device Minor:\s*(\d+)", contents)
        uuid_match = re.search(r"GPU UUID:\s*(\S+)", contents)
        if minor_match is None or uuid_match is None:
            raise ValueError(f"Missing GPU minor or UUID in {path}")
        minor = int(minor_match.group(1))
        if minor in allocated:
            if minor in mapped:
                raise ValueError(f"Duplicate NVIDIA device minor: {minor}")
            mapped[minor] = uuid_match.group(1).removeprefix("GPU-").lower()
    if set(mapped) != allocated or len(set(mapped.values())) != 6:
        raise ValueError(f"Incomplete GPU allocation mapping: {mapped!r}")
    return set(mapped.values())


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.device_count() != 6 or local_rank not in range(6):
        raise ValueError("The allocation must expose exactly six GPUs")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        expected_nccl_sum = world_size * (world_size + 1) / 2
        if world_size != 6:
            raise ValueError("The health probe requires exactly six ranks")
        device = torch.cuda.get_device_properties(local_rank)
        devices = [None] * world_size
        dist.all_gather_object(
            devices,
            {
                "rank": rank,
                "local_rank": local_rank,
                "host": socket.gethostname(),
                "name": device.name,
                "memory_bytes": device.total_memory,
                "uuid": str(device.uuid),
            },
        )
        if len({entry["host"] for entry in devices}) != 1:
            raise ValueError("All six GPUs must be allocated on one node")
        if {entry["local_rank"] for entry in devices} != set(range(6)):
            raise ValueError("Each local GPU must have exactly one probe rank")
        for entry in devices:
            if not any(kind in entry["name"] for kind in ("A100", "H100")):
                raise ValueError(f"Expected A100 or H100: {entry!r}")
            if entry["memory_bytes"] < 75 * 1024**3:
                raise ValueError(f"Expected at least 75 GiB per GPU: {entry!r}")
        expected = allocated_uuids()
        actual = {
            entry["uuid"].removeprefix("GPU-").lower() for entry in devices
        }
        if len(actual) != 6 or actual != expected:
            raise ValueError(
                f"GPU allocation UUID mismatch: expected {expected}, actual {actual}"
            )
        left = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16)
        right = torch.randn(
            (2048, 2048), device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        value = (left @ right).float().square().mean()
        value.backward()
        signal = torch.tensor([float(rank + 1)], device="cuda")
        dist.all_reduce(signal)
        torch.cuda.synchronize()
        passed = bool(
            torch.isfinite(value)
            and torch.isfinite(right.grad).all()
            and signal.item() == expected_nccl_sum
        )
        result = torch.tensor([int(passed)], device="cuda")
        dist.all_reduce(result, op=dist.ReduceOp.MIN)
        all_passed = bool(result.item())
        if rank == 0:
            print(
                json.dumps(
                    {
                        "host": socket.gethostname(),
                        "world_size": world_size,
                        "devices": devices,
                        "bf16_backward": all_passed,
                        "nccl_sum": signal.item(),
                        "expected_nccl_sum": expected_nccl_sum,
                        "passed": all_passed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if not all_passed:
            raise SystemExit(1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
