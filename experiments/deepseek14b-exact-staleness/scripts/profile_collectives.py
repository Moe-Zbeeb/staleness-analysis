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
    parser.add_argument("--transport", choices=["socket", "peer"], required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--group-size", type=int, default=4)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    size = args.group_size
    if size < 2 or world % size or torch.cuda.device_count() != world:
        raise ValueError("Collective groups must cover every allocated GPU exactly once")
    expected = "1" if args.transport == "socket" else "0"
    if any(os.environ.get(key) != expected for key in ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE")):
        raise ValueError("Set both transport variables before starting the process")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    groups = [dist.new_group(list(range(start, start + size))) for start in range(0, world, size)]
    group = groups[rank // size]
    rows = []
    for dtype in (torch.bfloat16, torch.float32):
        for size_mib in (16, 64, 256):
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
                        if not torch.all(full.chunk(size)[index] == rank // size * size + index + 1).item():
                            raise AssertionError("Incorrect gathered data")
                elif not torch.all(
                    reduced == sum(range(rank // size * size + 1, rank // size * size + size + 1))
                ).item():
                    raise AssertionError("Incorrect reduced data")
                row = {
                    "operation": name,
                    "dtype": str(dtype),
                    "full_tensor_mib": size_mib,
                    "median_seconds": statistics.median(times),
                    "seconds": times,
                    "algorithm_gb_s": size_mib * 1024 * 1024 / statistics.median(times) / 1e9,
                }
                rows.append(row)
                if rank % size == 0:
                    print(json.dumps({"transport": args.transport, "group": rank // size, **row}), flush=True)
            del shard, full, source, reduced
    if rank % size == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        result = {
            "transport": args.transport,
            "node": os.environ.get("SLURMD_NODENAME"),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "group_size": size,
            "physical_devices": os.environ["CUDA_VISIBLE_DEVICES"].split(",")[rank : rank + size],
            "torch": torch.__version__,
            "nccl": torch.cuda.nccl.version(),
            "correctness_passed": True,
            "rows": rows,
        }
        (args.output / f"{args.transport}-group-{rank // size}.json").write_text(json.dumps(result, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
