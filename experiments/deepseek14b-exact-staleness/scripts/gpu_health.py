import argparse
import importlib
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpus", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    count = torch.cuda.device_count()
    if world != args.expected_gpus or count != args.expected_gpus:
        raise RuntimeError(f"Expected {args.expected_gpus} GPUs and ranks; found {count} GPUs and {world} ranks")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != count or len(set(visible)) != count:
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match the allocation")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    try:
        signal = torch.tensor([local_rank + 1], device="cuda", dtype=torch.float32)
        dist.all_reduce(signal)
        expected = world * (world + 1) / 2
        if signal.item() != expected:
            raise RuntimeError("NCCL all-reduce produced an incorrect result")
        matrix = torch.nn.Linear(128, 128, device="cuda", dtype=torch.bfloat16)
        inputs = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        loss = matrix(inputs).float().square().mean()
        loss.backward()
        if not torch.isfinite(matrix.weight.grad).all().item():
            raise RuntimeError("BF16 backward produced nonfinite gradients")
        flash_attn = importlib.import_module("flash_attn")
        attention_inputs = [
            torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
        ]
        attention_output = flash_attn.flash_attn_func(*attention_inputs, causal=True)
        attention_output.float().square().mean().backward()
        if not all(torch.isfinite(value.grad).all().item() for value in attention_inputs):
            raise RuntimeError("Flash Attention backward produced nonfinite gradients")
        importlib.import_module("vllm")
        importlib.import_module("vllm._C_stable_libtorch")
        normalized = torch.empty_like(inputs)
        weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
        torch.ops._C.rms_norm(normalized, inputs, weight, 1e-6)
        reference = inputs.float() * torch.rsqrt(inputs.float().square().mean(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(normalized.float(), reference, rtol=0.02, atol=0.01)
        properties = torch.cuda.get_device_properties(local_rank)
        hardware = {
            "rank": local_rank,
            "visible_device": visible[local_rank],
            "name": properties.name,
            "bytes": properties.total_memory,
            "capability": torch.cuda.get_device_capability(local_rank),
            "all_reduce": signal.item(),
            "bf16_backward": True,
            "flash_attention_backward": True,
            "vllm_rms_norm": True,
        }
        gathered = [None] * world
        dist.all_gather_object(gathered, hardware)
        if local_rank == 0:
            receipt = {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": os.environ.get("SLURMD_NODENAME"),
                "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
                "world_size": world,
                "expected_all_reduce": expected,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "devices": gathered,
            }
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            with args.receipt.open("x") as stream:
                stream.write(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(receipt, indent=2), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
