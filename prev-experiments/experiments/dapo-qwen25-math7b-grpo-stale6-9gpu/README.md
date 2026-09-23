# Qwen2.5-Math-7B GRPO, staleness cap 6

Fresh 1,000-update run from the pinned Qwen2.5-Math-7B base, seed 42. Preserves the baseline dataset, tokenizer, terminal-answer reward, PPO-clipped GRPO objective, optimizer, batch size 64, group size 8, 4,096-token total context and 3,072 completion tokens.

Uses all nine A100 GPUs on deep-chungus-5: eight trainer GPUs and one inference GPU. The original cap-2 baseline used 4+2 and later 8+1; cap-4 uses 4+1. These execution-topology differences are recorded in comparison.json. The user explicitly authorized normal priority on all nine GPUs.

A separate 25-update smoke test validates collectives, BF16 backward, finite metrics, effective rollout ages, paired checkpoints and evaluation coverage. Only a passed smoke allows a fresh production run. All nine GPUs are checked against Slurm's allocated UUID set. Production metrics and evaluations stream to the separate Comet/Opik project qwen25-math7b-grpo-stale6-9gpu. Checkpoints have a projects/models link to persistent NFS outputs.
