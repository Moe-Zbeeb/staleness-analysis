import argparse
import json
import os
import socket
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from gpu_devices import GPU_COUNT, normalize_uuid, read_mapping, require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    args = parser.parse_args()
    mapping, mapping_sha = read_mapping(args.mapping)
    local_rank = int(os.environ["LOCAL_RANK"])
    require(torch.cuda.device_count() == GPU_COUNT, "The allocation must expose exactly five CUDA GPUs")
    require(0 <= local_rank < GPU_COUNT, "Unexpected local rank")
    device = torch.cuda.get_device_properties(local_rank)
    expected_uuid = mapping["allocated_devices"][local_rank]["uuid"]
    actual_uuid = normalize_uuid(str(device.uuid))
    require(actual_uuid == expected_uuid, f"GPU identity mismatch at local rank {local_rank}: expected {expected_uuid}, actual {actual_uuid}")
    require(any(kind in device.name for kind in ("A100", "H100")), f"Unsupported GPU: {device.name}")
    require(device.total_memory >= 75 * 1024**3, "Each allocated GPU must have at least 75 GiB")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        require(world_size == GPU_COUNT, "Health probe requires exactly five ranks")
        devices = [None] * world_size
        dist.all_gather_object(devices, {"rank": rank, "local_rank": local_rank, "host": socket.gethostname(), "name": device.name, "memory_bytes": device.total_memory, "uuid": actual_uuid})
        require(len({entry["host"] for entry in devices}) == 1, "All GPUs must be on one node")
        require({entry["local_rank"] for entry in devices} == set(range(GPU_COUNT)), "Local ranks must cover the allocation exactly once")
        require({entry["uuid"] for entry in devices} == {entry["uuid"] for entry in mapping["allocated_devices"]}, "Collective participants differ from allocated GPUs")
        left = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16)
        right = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        value = (left @ right).float().square().mean()
        value.backward()
        expected_sum = world_size * (world_size + 1) / 2
        signal = torch.tensor([float(rank + 1)], device="cuda")
        dist.all_reduce(signal)
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(value) and torch.isfinite(right.grad).all())
        result = torch.tensor([int(finite and signal.item() == expected_sum)], device="cuda")
        dist.all_reduce(result, op=dist.ReduceOp.MIN)
        passed = bool(result.item())
        if rank == 0:
            print(json.dumps({"host": socket.gethostname(), "world_size": world_size, "devices": devices, "mapping_sha256": mapping_sha, "cuda_visible_devices": mapping["cuda_visible_devices"], "bf16_backward": finite, "nccl_sum": signal.item(), "expected_nccl_sum": expected_sum, "passed": passed}, sort_keys=True), flush=True)
        require(passed, "GPU BF16 or NCCL collective validation failed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
