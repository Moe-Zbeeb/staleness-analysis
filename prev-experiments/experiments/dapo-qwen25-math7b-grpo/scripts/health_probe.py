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
        and signal.item() == 21.0
    )
    result = torch.tensor([int(passed)], device="cuda")
    dist.all_reduce(result, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(
            json.dumps(
                {
                    "host": socket.gethostname(),
                    "world_size": dist.get_world_size(),
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
