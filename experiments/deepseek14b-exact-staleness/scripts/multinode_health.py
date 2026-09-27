import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 12 or torch.cuda.device_count() != 6:
        raise ValueError("Cross-node preflight requires exactly twelve ranks and six GPUs per node")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    try:
        value = torch.tensor([rank + 1], dtype=torch.float32, device="cuda")
        dist.all_reduce(value)
        if value.item() != 78:
            raise RuntimeError("Cross-node NCCL collective returned an incorrect result")
        tensor = torch.ones(8 * 1024 * 1024, dtype=torch.float32, device="cuda")
        dist.barrier()
        torch.cuda.synchronize()
        start = time.monotonic()
        for _ in range(5):
            dist.broadcast(tensor, src=0)
        torch.cuda.synchronize()
        seconds = time.monotonic() - start
        if rank == 0:
            args.receipt.write_text(
                json.dumps(
                    {
                        "world_size": 12,
                        "all_reduce": value.item(),
                        "broadcast_bytes": 5 * tensor.numel() * tensor.element_size(),
                        "broadcast_seconds": seconds,
                    },
                    indent=2,
                )
                + "\n"
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
