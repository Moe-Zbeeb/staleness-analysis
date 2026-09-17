# Qwen2.5-Math-1.5B GRPO, staleness cap 8, five GPUs

User-requested normal-QoS run on the remaining five A100 GPUs of deep-chungus-9, using three trainer GPUs and two inference GPUs. Runs alongside the cap-6 job without a dependency. Account grad-students, partition low-priority, QoS normal. The job is requeue-enabled and can be preempted.

Fresh 1,000-step production run from the original pinned base model, seed 42, batch size 64, group size 8, and 3,072 completion tokens. Scientific settings match the cap-4 setup except the inclusive staleness cap. GPU topology also changes and must be accounted for in comparisons.

A separate 25-step smoke test validates the exact configuration before production. All five allocated GPUs are used. Source hashes, data/tokenizer identity, finite losses and gradients, actual rollout ages, runtime settings, paired checkpoints, and final evaluation coverage are checked.
