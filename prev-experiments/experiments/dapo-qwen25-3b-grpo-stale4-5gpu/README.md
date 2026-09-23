# Qwen2.5-3B GRPO, staleness cap 4

Fresh 1,000-update run from the pinned Qwen2.5-3B base model, seed 42. This uses the original 3B baseline data, tokenizer, optimizer, reward and PPO-clipped GRPO objective, batch size 64, group size 8, and 3,072 completion tokens.

The inclusive rollout staleness cap is 4. Run identity, checkpoints and Comet/Opik project are isolated under `qwen25-3b-grpo-stale4-5gpu`. Production telemetry starts after smoke validation.

The user requested deep-chungus-7. The user requested normal priority on the five available A100 GPUs, with grad-students / low-priority / normal: three trainer GPUs and two inference GPUs. The historical cap-2 baseline used four trainer plus three inference GPUs, so topology is an additional comparison difference. The job shares the node with existing allocations. Normal-priority execution can be preempted; the wrapper supports same-arm checkpoint recovery.

The wrapper performs a separate 25-update smoke, validates resolved configuration, effective rollout ages, finite metrics and paired checkpoints, then starts fresh 1,000-update production. It resumes only this same arm after an interruption. Source, data and tokenizer hashes must match the frozen manifests. Model artifacts live under projects/models with persistent storage links.
