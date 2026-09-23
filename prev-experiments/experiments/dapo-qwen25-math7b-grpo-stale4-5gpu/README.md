# Qwen2.5-Math-7B GRPO, staleness cap 4

Fresh 1,000-update run from pinned Qwen2.5-Math-7B, seed 42. Preserves the original 7B dataset, tokenizer, reward, PPO-clipped GRPO objective, optimizer, batch size 64, group size 8, 4,096-token context and 3,072 completion tokens.

Uses the five free A100 GPUs requested on deep-chungus-7: four trainer GPUs and one inference GPU. The baseline used 4+2 and later 8+1, so execution topology also differs. The user explicitly authorized normal priority on these five GPUs.

A separate 25-update smoke validates GPU collectives, BF16 backward, finite training metrics, effective rollout ages, paired checkpoints and evaluation coverage before fresh production begins. Comet/Opik telemetry uses qwen25-math7b-grpo-stale4-5gpu. Checkpoints have a projects/models link to persistent NFS outputs.
