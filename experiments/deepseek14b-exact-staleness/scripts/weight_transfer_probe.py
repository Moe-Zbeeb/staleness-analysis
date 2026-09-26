import argparse
import json
import os
from datetime import timedelta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["upstream", "startup", "socket", "peer"], required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    allocated = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if int(os.environ["WORLD_SIZE"]) != 8 or len(allocated) != 8:
        raise ValueError("Weight-transfer probe requires all eight allocated GPUs")
    device = rank if rank < 4 else rank - 4
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(allocated[:4] if rank < 4 else allocated[4:])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,ENV,GRAPH"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:" + ("False" if rank < 4 else "True")
    if args.transport in {"socket", "peer"}:
        value = "1" if args.transport == "socket" else "0"
        os.environ["NCCL_P2P_DISABLE"] = value
        os.environ["NCCL_SHM_DISABLE"] = value
    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable

    if torch.cuda.device_count() != 4:
        raise ValueError("Each side of the weight-transfer probe must see exactly four GPUs")
    torch.cuda.set_device(device)
    if args.transport == "startup":
        disable_nccl_p2p_if_unavailable()
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    for members in ([0], [1], [2], [3], [4, 5, 6, 7]):
        group = dist.new_group(members, backend="nccl", timeout=timedelta(seconds=60))
        if rank in members:
            value = torch.tensor([rank + 1.0], device="cuda")
            dist.all_reduce(value, group=group)
            assert value.item() == sum(member + 1 for member in members)
    dist.barrier()
    if rank <= 4:
        disable_nccl_p2p_if_unavailable()
        bridge_rank = 0 if rank == 4 else rank + 1
        group = StatelessProcessGroup.create(
            host="127.0.0.1", port=29751, rank=bridge_rank, world_size=5, store_timeout=60
        )
        communicator = PyNcclCommunicator(group, device=torch.device("cuda", device))
        for size in (1024, 1048576):
            tensor = torch.full((size,), 7.0 if bridge_rank == 0 else 0.0, dtype=torch.bfloat16, device="cuda")
            communicator.broadcast(tensor, src=0)
            torch.cuda.synchronize()
            assert torch.all(tensor == 7).item()
        print(json.dumps({"transport": args.transport, "rank": rank, "bridge_rank": bridge_rank, "passed": True}), flush=True)
    dist.barrier()
    if rank == 0:
        print(json.dumps({"transport": args.transport, "world_size": 8, "status": "passed"}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
