import io
import random

import numpy as np
import torch

from deepseek_study.checkpoints import atomic_write


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": {"device": torch.cuda.current_device(), "state": torch.cuda.get_rng_state()}
        if torch.cuda.is_available()
        else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if state["cuda"]["device"] != torch.cuda.current_device():
            raise ValueError("CUDA RNG state belongs to a different trainer device layout")
        torch.cuda.set_rng_state(state["cuda"]["state"])


class CheckpointWithRNG:
    def __init__(self, manager):
        self.manager = manager

    def __getattr__(self, name):
        return getattr(self.manager, name)

    def maybe_clean(self):
        return None

    def save(self, step, *args, **kwargs):
        self.manager.save(step, *args, **kwargs)
        rank = self.manager.world.rank
        state = {"rank": rank, "world_size": torch.distributed.get_world_size(), "rng": capture_rng()}
        buffer = io.BytesIO()
        torch.save(state, buffer)
        atomic_write(self.manager.get_ckpt_path(step).parent / "rng" / f"rank_{rank}.pt", buffer.getvalue())
        torch.distributed.barrier()

    def load(self, step, *args, **kwargs):
        self.manager.load(step, *args, **kwargs)
        path = kwargs.get("path") or self.manager.get_ckpt_path(step)
        rank = self.manager.world.rank
        state = torch.load(path.parent / "rng" / f"rank_{rank}.pt", weights_only=False, map_location="cpu")
        if state["rank"] != rank or state["world_size"] != torch.distributed.get_world_size():
            raise ValueError("Trainer RNG checkpoint requires the original sharding layout")
        restore_rng(state["rng"])
