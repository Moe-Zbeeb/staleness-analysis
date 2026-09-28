# DAPO 1.5B, exact k256, 32 responses per update

The user requested a smaller optimizer batch while preserving all other settings and the exact-staleness algorithm. This replaces learner 2145435 and historical worker 2145436, both cancelled on September 28. Their outputs remain intact. The old run completed one update in 2,232.95 seconds, with 1,949.22 seconds waiting for learner publication and 39.92 seconds applying weights; its first checkpoint has a shared `backup-verified.json` receipt.

## New jobs

| Role | Job | Node | GPUs | Priority |
| --- | --- | --- | --- | --- |
| Learner and current inference | 2145464 | deep-chungus-5 | 9 A100 40GB: 4 trainer + 5 inference | High |
| Bootstrap historical inference | 2145465 | deep-chungus-3 | 3 A100 40GB | High |

Both allocations started. Startup validation and the first completed optimizer update must be checked before drawing throughput conclusions. The historical job depends on the learner starting, not finishing. The main allocation remains exclusive, with 96 CPUs and a 45-day limit; the worker has 48 CPUs, 160 GiB requested memory and a 14-day limit. Automatic requeue remains disabled.

The [submission receipt](submission.json) contains the exact commands and initial Slurm verification. Its initial pending states are submission-time snapshots. [The configuration diff](batch-change.json) records every changed field against the previous frozen deployment.

## The single training-setting change

`prompts_per_update` changes from **64 to 4**. `responses_per_prompt` remains **8**, giving **32 responses per optimizer update**, rather than 512. This is the global batch, not the per-GPU microbatch or inference concurrency. The latter settings are unchanged, including the 512 dispatcher concurrency ceiling; the finite source dispatches only the requested four complete groups per cohort.

Everything else in [study.json](study.json) is retained: pinned DeepSeek-R1-Distill-Qwen-1.5B, the same 17,005-row DAPO dataset and order, 6,144 response-token cap, 2,048 prompt-token cap, seed 42, GRPO clipping 0.2, centered group advantages, global token normalization, zero KL and weight decay, LR 1e-6 with 30-update warmup, FP32 optimizer/reductions, activation checkpointing, disabled forward resharding, disabled compilation and the same GPU topology. NCCL environment settings are intentionally unchanged for this comparison.

The run starts from the original model with a fresh optimizer and queue. It does not resume the 512-response checkpoint or reuse that run's historical batches. At 1,000 updates, the consumed response budget is now 32,000 instead of 512,000. Lag remains measured in optimizer updates; equal k and update count do not imply equal training-token exposure. With four questions per update, this run consumes the first 4,000 positions of the same prompt stream.

## Algorithm and persistence

Updates 1–256 use on-policy cohorts. During bootstrap, the remote worker generates separate future cohorts from immutable policies theta_0 through theta_255. Update 257 consumes theta_0 data while the learner is theta_256. Every subsequent consumed batch has exact age 256; a missing batch causes a wait, never substitution. After bootstrap the main inference pool generates future batches while the learner consumes older ones. The final 256 updates drain the queue.

Checkpoints remain at update 1, every 25 updates and completion; local retention and verified NFS backup retention are unchanged. Metrics continue to XFS and TensorBoard, with no Runboard or intermediate evaluations. The batch change also applies to the historical worker's cohort contract.

All source files are byte-identical to the previous frozen release (`cb27c5b`), with package and control hashes checked before submission. Official PrimeRL and vLLM source are unmodified. Only deployment configuration and unique paths differ; the dataset and prepared manifest bytes are copied unchanged.

## Cluster paths

- Controls: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek15b-dapo17k-async-b32-hp-20260928-v1`
- NFS output: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/outputs/deepseek15b-dapo17k-6k-exact256-b32-seed42-v1`
- Main active workspace: `/tmp/mohamadzbib-staleness-storage-v2/deepseek15b-dapo17k-6k-exact256-b32-seed42-v1`
- Historical queue: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/historical-rollouts/deepseek15b-dapo17k-6k-exact256-b32-seed42-v1`
- XFS metrics: `/mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics/deepseek15b-dapo17k-6k-exact256-b32-seed42-v1`

The local TensorBoard viewer uses a separate event directory for this run to prevent mixed-batch curves.
