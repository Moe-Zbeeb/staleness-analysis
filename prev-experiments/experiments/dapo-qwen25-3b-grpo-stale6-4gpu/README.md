# Qwen2.5-3B GRPO, staleness cap 6, four GPUs

Fresh 1,000-update production run from the pinned Qwen2.5-3B base model, seed 42. This replaces the never-started six-GPU cap-6 job 2141819 on deep-chungus-10 at the user's explicit request.

Use exactly four free A100 GPUs on deep-chungus-9: two trainer GPUs plus two inference GPUs. Scheduling is grad-students / low-priority / normal, with 32 CPUs and 160 GiB host memory, without exclusive-node allocation. The user explicitly requested no HP so this run can start immediately. Normal-priority preemption remains possible; same-arm resume and requeue are enabled.

Preserve the baseline data, tokenizer, reward, PPO-clipped GRPO objective, batch size 64, group size 8, learning rate 1e-6, 4,096 total context and 3,072 completion-token cap. The inclusive rollout staleness bound is six. Historical cap-2 used four trainer plus three inference GPUs, so topology is an additional comparison difference.

The wrapper materializes and verifies frozen baseline assets, validates runtime configuration, checks all four GPUs, runs a separate 25-update smoke with finite-metric and checkpoint checks, and then starts fresh production. Training and checkpoint recovery use the isolated four-GPU run identity. Comet/Opik project: qwen25-3b-grpo-stale6-4gpu. Model artifacts remain under projects/models through persistent storage links. Explicit inference and transport ports isolate this run from other jobs on the shared node.
