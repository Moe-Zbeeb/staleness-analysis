import os
import random

import numpy as np
import torch


def main():
    seed = int(os.environ["DEEPSEEK_STUDY_SEED"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    from prime_rl.trainer.rl import train
    from deepseek_study.runtime.trainer_state import CheckpointWithRNG

    setup = train.setup_ckpt_manager

    def setup_with_rng(*args, **kwargs):
        return CheckpointWithRNG(setup(*args, **kwargs))

    train.setup_ckpt_manager = setup_with_rng
    try:
        train.main()
    finally:
        train.setup_ckpt_manager = setup


if __name__ == "__main__":
    main()
