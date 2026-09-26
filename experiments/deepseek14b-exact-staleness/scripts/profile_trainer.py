import argparse
import functools
import hashlib
import json
import os
import random
import time
from pathlib import Path
from types import MethodType

import msgspec
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micro-batches", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.micro_batches < 1 or not 1 <= args.steps <= 10:
        raise ValueError("Profiling requires positive micro-batches and at most ten updates")
    rank = int(os.environ["LOCAL_RANK"])
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    from prime_rl.configs.trainer import TrainerConfig
    from prime_rl.trainer.rl import train
    from prime_rl.trainer.rl.data import DataLoader
    from prime_rl.transports.batch import MicroBatch
    from deepseek_study.tracking.tokens import PaperTokenExporter

    output = args.output / f"rank-{rank}"
    output.mkdir(parents=True, exist_ok=False)
    grid_bytes = args.grid.read_bytes()
    grid = msgspec.msgpack.decode(grid_bytes, type=list[list[MicroBatch]])
    if len(grid) != int(os.environ["WORLD_SIZE"]):
        raise ValueError("Replay grid must match the original trainer world size")
    selected = grid[rank][: args.micro_batches]
    if len(selected) != args.micro_batches:
        raise ValueError("Insufficient archived micro-batches")
    batches = [DataLoader._micro_batch_to_tensor(None, item) for item in selected]
    del grid, selected
    rows = []
    state = {"step": 0, "profiler": None}

    def timed(name, function):
        @functools.wraps(function)
        def wrapped(*positional, **keywords):
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.profiler.record_function(name):
                result = function(*positional, **keywords)
            torch.cuda.synchronize()
            rows.append({"step": state["step"], "phase": name, "seconds": time.perf_counter() - started})
            return result

        return wrapped

    class Replay:
        def __init__(self, *positional, **keywords):
            pass

        def wait_for_batch(self):
            return

        def get_batch(self):
            state["step"] += 1
            if args.trace and state["step"] == 2:
                state["profiler"] = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                )
                state["profiler"].start()
            return batches

    class TimedExporter(PaperTokenExporter):
        def __init__(self, config, parallel_dims, world, logger):
            super().__init__(config.output_dir, world.rank, config.loss.kwargs["clip_epsilon"])
            self.export = timed("token_export", self.export)
            self.mark_stable = timed("token_compression_and_write", self.mark_stable)

    original_optimizer = train.setup_optimizer

    def setup_optimizer(*positional, **keywords):
        optimizer, manager = original_optimizer(*positional, **keywords)
        original_step = optimizer.step

        @functools.wraps(original_step)
        def step(self, *step_args, **step_kwargs):
            result = timed("optimizer", original_step)(*step_args, **step_kwargs)
            if state["profiler"] is not None:
                profiler = state["profiler"]
                profiler.stop()
                events = profiler.key_averages()
                summary = [
                    {
                        "name": event.key,
                        "count": event.count,
                        "self_cpu_us": event.self_cpu_time_total,
                        "self_device_us": event.self_device_time_total,
                    }
                    for event in events
                ]
                (output / "operators.json").write_text(json.dumps(summary, indent=2) + "\n")
                profiler.export_chrome_trace(str(output / "trace.json.gz"))
                state["profiler"] = None
            return result

        optimizer.step = MethodType(step, optimizer)
        return optimizer, manager

    config_data = json.loads(args.config.read_text())
    config_data.update(
        output_dir=str(args.output / "trainer"),
        max_steps=args.steps,
        ckpt=None,
        resume=None,
        data={"fake": {"batch_size": len(batches) * len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))}},
        monitors={"file": {}},
        trace_path=None,
        heartbeat=None,
        metrics_server=None,
    )
    config_data["log"]["level"] = "debug"
    config = TrainerConfig.model_validate(config_data)
    (output / "config.json").write_text(config.model_dump_json(indent=2) + "\n")
    train.FakeDataLoader = Replay
    train.setup_token_exporter = TimedExporter
    train.setup_optimizer = setup_optimizer
    train.forward = timed("forward", train.forward)
    train.compute_loss = timed("loss", train.compute_loss)
    original_backward = torch.Tensor.backward
    torch.Tensor.backward = timed("backward", original_backward)
    try:
        train.train(config)
    finally:
        torch.Tensor.backward = original_backward
        (output / "timings.json").write_text(
            json.dumps(
                {
                    "grid_sha256": hashlib.sha256(grid_bytes).hexdigest(),
                    "node": os.environ.get("SLURMD_NODENAME"),
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                    "rank": rank,
                    "transport": {key: os.environ.get(key) for key in ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE")},
                    "micro_batches": len(batches),
                    "steps": args.steps,
                    "phase_synchronization": True,
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
