# Qwen2.5-Math-1.5B GRPO, staleness cap 4, three GPUs

User-authorized fresh 1,000-update run from the pinned original base model. Uses two trainer GPUs and one inference GPU on a shared A100 node, account/partition/QoS `grad-students` / `high-priority` / `high-priority`.

The historical cap-2 run used three trainer GPUs and two inference GPUs. GPU subtype and sharing are recorded at startup. This is not a comparison in which only staleness changes.

A separate 25-update smoke run preserves production batch size 64, group size 8, 3,072 completion tokens, and the 30-step learning-rate warmup. Production starts from the original base model after smoke acceptance; smoke weights are never resumed into production. The cap is inclusive: `(shipping step - 1) - policy.start <= 4`.

Source hashes, copied dataset/tokenizer hashes, runtime settings, finite loss and gradient, effective rollout provenance, all allocated GPUs, and paired checkpoints are validated. A process lock protects the arm across smoke, training, and same-arm recovery. Output and observer state are separate from every previous run.

Production checkpoint link: `/mnt/xfs/home/mohamadzbib/projects/models/qwen25-math15b-grpo-seed42-stale4-3gpu`.

Intermediate evaluation labels identify trigger steps and may span live policy updates. Use frozen checkpoint evaluation when strict checkpoint attribution is required.
