# Qwen2.5-Math-7B GRPO, staleness cap 6 on eight H100s

Fresh 1,000-update run from the pinned base model, seed 42. Uses all eight H100 GPUs on deep-h-3: four trainer GPUs and four inference replicas. The user explicitly authorized normal priority. This replaces A100 job 2141902, which was preempted before producing any smoke or production artifacts.

Dataset, tokenizer, reward, optimizer, PPO clipping, batch size 64, group size 8, 4,096-token context and 3,072-token training completion budget remain fixed. Hardware and GPU topology changes are recorded in comparison.json.

Bounded NVIDIA and all-eight-GPU BF16/NCCL probes run before model staging. A separate 25-update smoke validates effective staleness, finite updates, paired checkpoints and evaluation coverage before fresh production starts. Comet project: qwen25-math7b-grpo-stale6-8h100. Checkpoints are linked under projects/models to persistent NFS outputs.
