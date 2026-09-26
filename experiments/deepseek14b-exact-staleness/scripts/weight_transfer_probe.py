import argparse
import json
import os
from datetime import timedelta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["upstream", "startup", "network", "socket", "peer"], required=True)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--inference-gpus", type=int, default=4)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    allocated = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    count, inference = args.expected_gpus, args.inference_gpus
    if int(os.environ["WORLD_SIZE"]) != count or len(allocated) != count or len(set(allocated)) != count:
        raise ValueError("Weight-transfer probe requires the complete GPU allocation")
    if not 1 <= inference < count or not 0 <= rank < count:
        raise ValueError("Every GPU must belong to exactly one trainer or inference process")
    device = rank if rank < inference else rank - inference
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(allocated[:inference] if rank < inference else allocated[inference:])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,ENV,GRAPH"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:" + ("False" if rank < inference else "True")
    if args.transport in {"network", "socket", "peer"}:
        value = "0" if args.transport == "peer" else "1"
        os.environ["NCCL_P2P_DISABLE"] = value
        os.environ["NCCL_SHM_DISABLE"] = value
    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable

    expected_visible = inference if rank < inference else count - inference
    if torch.cuda.device_count() != expected_visible:
        raise ValueError("Each side must see exactly its assigned GPUs")
    torch.cuda.set_device(device)
    if args.transport == "startup":
        disable_nccl_p2p_if_unavailable()
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    for members in [[index] for index in range(inference)] + [list(range(inference, count))]:
        group = dist.new_group(members, backend="nccl", timeout=timedelta(seconds=60))
        if rank in members:
            value = torch.tensor([rank + 1.0], device="cuda")
            dist.all_reduce(value, group=group)
            assert value.item() == sum(member + 1 for member in members)
    dist.barrier()
    if rank <= inference:
        disable_nccl_p2p_if_unavailable()
        bridge_rank = 0 if rank == inference else rank + 1
        group = StatelessProcessGroup.create(
            host="127.0.0.1", port=29751, rank=bridge_rank, world_size=inference + 1, store_timeout=60
        )
        communicator = PyNcclCommunicator(group, device=torch.device("cuda", device))
        for size in (1024, 1048576):
            tensor = torch.full((size,), 7.0 if bridge_rank == 0 else 0.0, dtype=torch.bfloat16, device="cuda")
            communicator.broadcast(tensor, src=0)
            torch.cuda.synchronize()
            assert torch.all(tensor == 7).item()
        print(
            json.dumps({"transport": args.transport, "rank": rank, "bridge_rank": bridge_rank, "passed": True}),
            flush=True,
        )
    dist.barrier()
    if rank == 0:
        print(
            json.dumps(
                {
                    "transport": args.transport,
                    "world_size": count,
                    "inference_gpus": inference,
                    "trainer_gpus": count - inference,
                    "status": "passed",
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
