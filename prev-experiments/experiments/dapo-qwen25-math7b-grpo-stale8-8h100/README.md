# Qwen2.5-Math-7B GRPO, staleness cap 8 on eight H100s

Fresh 1,000-update run from the pinned base model, seed 42. Uses all eight H100 GPUs on deep-h-1: four trainer GPUs and four inference replicas, continuing the user-requested normal-priority H100 setup.

Dataset, tokenizer, reward, optimizer, PPO clipping, batch size 64, group size 8, 4,096-token context and 3,072-token training completion budget remain fixed. Hardware and GPU topology changes from the cap-2 baseline are recorded in comparison.json. The separate cap-6 H100 run uses the same topology.

Bounded NVIDIA and all-eight-GPU BF16/NCCL probes run before model staging. A separate 25-update smoke validates effective staleness, finite updates, paired checkpoints and evaluation coverage before fresh production starts. Comet project: qwen25-math7b-grpo-stale8-8h100. Checkpoints are linked under projects/models to persistent NFS outputs.
