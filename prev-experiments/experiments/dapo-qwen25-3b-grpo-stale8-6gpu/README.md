# Qwen2.5-3B GRPO, staleness cap 8

Fresh 1,000-update run from the pinned Qwen2.5-3B base model, seed 42. This preserves the original 3B data, tokenizer, optimizer, reward, PPO-clipped GRPO objective, batch size 64, group size 8, and 3,072 completion tokens.

The inclusive rollout staleness cap is 8. Run identity, checkpoints, recovery state and Comet/Opik project are isolated under `qwen25-3b-grpo-stale8-6gpu`.

The user requested the six available A100 GPUs on deep-chungus-10. The allocation uses four trainer GPUs and two inference GPUs, 64 CPUs and 160 GiB host memory, with grad-students / low-priority / normal and no exclusive-node request. All six GPUs must pass the BF16 backward and NCCL collective health checks. Device ordering is derived from the actual allocation, including non-contiguous physical GPU IDs.

During preparation another job occupied the six GPUs. This run is prepared to queue on deep-chungus-10 until sufficient node capacity becomes available, independently of the 14B job and high-priority GPU allowance. Normal-priority jobs can be preempted. Slurm wall time is unlimited; requeue and same-arm checkpoint recovery are enabled. The cap-6 and cap-8 arms each request six GPUs and consequently run one at a time on this eight-GPU node.

The historical cap-2 baseline used four trainer plus three inference GPUs; cap 4 used three plus two. Topology and node placement therefore also differ in comparisons with this arm.

The wrapper performs a separate 25-update smoke, verifies resolved configuration, rollout ages, finite metrics, evaluation coverage and paired checkpoints, then starts fresh 1,000-update production. Production telemetry starts only after smoke validation. Source, data and tokenizer hashes must match the frozen manifests. Model artifacts live under projects/models through persistent storage links.
