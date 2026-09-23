import json
import os
import socket

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if torch.cuda.device_count() != 6 or dist.get_world_size() != 6:
        raise ValueError("The allocation must expose exactly six GPUs")
    device = torch.cuda.get_device_properties(local_rank)
    if "A100" not in device.name or device.total_memory < 38 * 1024**3:
        raise ValueError("Expected A100 with at least 38 GiB per allocated device")
    devices = [None] * dist.get_world_size()
    dist.all_gather_object(devices, {"local_rank": local_rank, "name": device.name, "memory_bytes": device.total_memory})
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
        and signal.item() == dist.get_world_size() * (dist.get_world_size() + 1) / 2
    )
    result = torch.tensor([int(passed)], device="cuda")
    dist.all_reduce(result, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(
            json.dumps(
                {
                    "host": socket.gethostname(),
                    "world_size": dist.get_world_size(),
                    "devices": devices,
                    "device": torch.cuda.get_device_name(local_rank),
                    "bf16_backward": bool(result.item()),
                    "nccl_sum": signal.item(),
                },
                sort_keys=True,
            )
        )
    dist.destroy_process_group()
    if not result.item():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
