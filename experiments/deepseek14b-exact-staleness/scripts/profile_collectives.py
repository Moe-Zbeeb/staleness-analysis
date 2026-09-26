import argparse
import json
import os
import statistics
import time
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport", choices=["network", "peer"], required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--group-sizes", type=int, nargs="+")
    parser.add_argument("--sizes-mib", type=int, nargs="+", default=[16, 64, 256])
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if args.group_size < 2:
        raise ValueError("Collective groups need at least two ranks")
    sizes = args.group_sizes or [args.group_size] * (world // args.group_size)
    if min(sizes, default=0) < 2 or sum(sizes) != world or torch.cuda.device_count() != world:
        raise ValueError("Collective groups must cover every allocated GPU exactly once")
    if not 1 <= args.repeats <= 100 or any(not 1 <= value <= 2048 for value in args.sizes_mib):
        raise ValueError("Use bounded positive repeat counts and tensor sizes")
    expected = "1" if args.transport == "network" else "0"
    if any(os.environ.get(key) != expected for key in ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE")):
        raise ValueError("Set both transport variables before starting the process")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    offset = 0
    for index, size in enumerate(sizes):
        members = list(range(offset, offset + size))
        candidate = dist.new_group(members)
        if rank in members:
            group, group_index, group_start, group_size = candidate, index, offset, size
        offset += size
    size = group_size
    rows = []
    for dtype in (torch.bfloat16, torch.float32):
        for size_mib in args.sizes_mib:
            count = size_mib * 1024 * 1024 // torch.empty((), dtype=dtype).element_size()
            count = count // size * size
            shard = torch.full((count // size,), rank + 1, device="cuda", dtype=dtype)
            full = torch.empty(count, device="cuda", dtype=dtype)
            source = torch.full((count,), rank + 1, device="cuda", dtype=dtype)
            reduced = torch.empty_like(shard)
            operations = {
                "all_gather": partial(dist.all_gather_into_tensor, full, shard, group=group),
                "reduce_scatter": partial(dist.reduce_scatter_tensor, reduced, source, group=group),
            }
            for name, operation in operations.items():
                for _ in range(3):
                    operation()
                torch.cuda.synchronize()
                times = []
                for _ in range(args.repeats):
                    dist.barrier()
                    started = time.perf_counter()
                    operation()
                    torch.cuda.synchronize()
                    elapsed = torch.tensor([time.perf_counter() - started], device="cuda", dtype=torch.float64)
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
                    times.append(elapsed.item())
                if name == "all_gather":
                    for index in range(size):
                        if not torch.all(full.chunk(size)[index] == group_start + index + 1).item():
                            raise AssertionError("Incorrect gathered data")
                elif not torch.all(reduced == sum(range(group_start + 1, group_start + size + 1))).item():
                    raise AssertionError("Incorrect reduced data")
                row = {
                    "operation": name,
                    "dtype": str(dtype),
                    "full_tensor_mib": size_mib,
                    "full_tensor_bytes": count * torch.empty((), dtype=dtype).element_size(),
                    "median_seconds": statistics.median(times),
                    "seconds": times,
                    "algorithm_gb_s": count
                    * torch.empty((), dtype=dtype).element_size()
                    / statistics.median(times)
                    / 1e9,
                }
                rows.append(row)
                if rank == group_start:
                    print(json.dumps({"transport": args.transport, "group": group_index, **row}), flush=True)
            del shard, full, source, reduced
    if rank == group_start:
        args.output.mkdir(parents=True, exist_ok=True)
        result = {
            "transport": args.transport,
            "node": os.environ.get("SLURMD_NODENAME"),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "group_size": size,
            "physical_devices": os.environ["CUDA_VISIBLE_DEVICES"].split(",")[rank : rank + size],
            "torch": torch.__version__,
            "nccl": torch.cuda.nccl.version(),
            "nccl_environment": {key: value for key, value in os.environ.items() if key.startswith("NCCL_")},
            "correctness_passed": True,
            "rows": rows,
        }
        with (args.output / f"{args.transport}-group-{group_index}.json").open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
