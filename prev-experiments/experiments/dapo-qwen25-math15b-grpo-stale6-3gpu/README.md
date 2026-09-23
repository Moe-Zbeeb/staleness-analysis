# Qwen2.5-Math-1.5B GRPO, staleness cap 6

Fresh 1,000-update run from the original pinned base model, seed 42, matching the completed cap-4 setup: 2 trainer GPUs plus 1 inference GPU, batch size 64, group size 8, and 3,072 completion tokens. The only changed training configuration is the inclusive staleness cap (6); run identities are separate.

Runs on deep-chungus-9 using grad-students / high-priority / high-priority. Cap 8 queues after cap 6 to preserve the 12-GPU high-priority limit while the 9-GPU 14B job runs. Hardware placement differs from cap 4 and is recorded at startup.

A separate 25-update smoke test must pass before fresh production starts. Source, data, tokenizer, resolved configuration, GPU assignment, finite gradients, actual rollout ages, final evaluation coverage, and paired checkpoints are validated by the inherited guard. The historical cap-2 baseline used a different GPU topology.
