# DeepSeek 1.5B optimized fresh run

On September 27 the user authorized discarding the unfinished run and starting the faster candidate at high priority. Previous job `2145184` stopped intentionally after 13 committed updates; its local/shared output is retained for audit. Slurm records FAILED (exit 1) because the supervisor raises on termination; the recorded cause is the user-requested stop. The shared backup inventory was refreshed during shutdown. No checkpoint is imported into the replacement; all 1,000 updates start from the pinned initial model.

The replacement uses eight healthy A100 80 GB GPUs, four trainers and four inference workers. It excludes node 7, where an inference GPU was thermally throttled and another GPU failed health checks. Inference maximum active sequences changes from 16 to 64, dispatcher concurrency from 64 to 256, and `trainer_reshard_after_forward` from true to false. Activation checkpointing remains enabled and compilation remains disabled. Imported PrimeRL/vLLM source is unchanged.

The model, dataset/grader, response length, sampling settings, optimizer, LR schedule, GRPO objective, exact lag 256 schedule, 1,000-update budget and 100-update checkpoint interval are unchanged. TensorBoard is the only dashboard. Active files, runtime, assets and caches remain node-local, with verified NFS backups and XFS metric copies.

## Validation before full training

[`validate_then_run.py`](../../scripts/validate_then_run.py) stages and verifies the frozen release, runtime, tokenizer, model and dataset. It requires eight healthy 80 GB A100s and verifies eight-rank collectives, BF16 backward, Flash Attention backward and vLLM operations.

Within the same allocation, it runs two complete bootstrap updates for the baseline and two for the optimized candidate. Each case has a one-hour supervisor deadline. The cases share the same four-training/four-inference topology. Both must complete exactly two updates, preserve 512 responses per update and age zero during bootstrap, show positive rewards and nonzero advantages/gradient norms, and log finite training metrics. The candidate's mean update time must be lower than the baseline before the full run starts automatically. Failure exits the job and preserves diagnostics; there is no automatic retry or restart from scratch.

This is an operational readiness and timing check, not evidence of identical numerical trajectories or a statistically precise speedup estimate. Fixed seeds do not make different inference batch sizes bitwise equivalent. The earlier [matrix](../optimization-1p5b-20260927.md) measured these differences. The validation does not measure the age-256 phase or recovery checkpoint cost. Validation steps live in separate outputs and do not count toward the fresh full run.

## Storage

- Control: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek15b-full-hp-opt-20260927`
- Local work: `/tmp/staleness-storage-v2/mohamadzbib/deepseek15b-exact256-80gb-seed42-v3-opt`
- Local runtime: `/tmp/staleness-runtime/mohamadzbib/deepseek15b-opt-v3`
- Shared backup: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/outputs/deepseek15b-exact256-80gb-seed42-v3-opt`
- Metric copy: `/mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics/deepseek15b-exact256-80gb-seed42-v3-opt`

The previous two-update diagnostic `2145250` is superseded by the validation in this full-run allocation. Replacement job `2145258` requests account `grad-students`, partition/QoS `high-priority`, one exclusive eight-A100 node, 128 CPU slots and a 45-day wall limit, without automatic requeue. Eligible GPU nodes are 9, 10 and 11; nodes 1–8 and the older GPU nodes are excluded. It was submitted held, checked, and released after the old run received a graceful termination signal. A scheduler allocation does not by itself establish successful validation or training startup.

At submission, job `2145258` was pending for resources after the old GPU quota was released. Slurm estimated October 3, which is provisional. The user was offered immediate validation on node 7 with its seven working GPUs or waiting for a healthy eight-GPU node; the submitted job retains the eight-GPU requirement until a decision is made.
