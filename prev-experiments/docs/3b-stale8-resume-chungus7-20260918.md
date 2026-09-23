# 3B staleness ≤8 relocation — September 18, 2026

The user requested finishing the queued 3B staleness ≤8 run on deep-chungus-7 at normal priority, then explicitly chose it ahead of the 7B staleness ≤4 run after that run reclaimed the node's available GPUs.

## Scheduling

- Existing 3B job: 2141858, account grad-students, partition low-priority, QoS normal.
- Placement changed from deep-chungus-9 to deep-chungus-7 using `scontrol update`, preserving the job ID and batch script.
- Allocation remains four A100 GPUs, two trainer workers and two inference workers, 32 CPUs and 160G requested host memory.
- 7B job 2141895 was requeued with a hold, given dependency `afterany:2141858`, then released. It becomes eligible after the 3B job ends and remains subject to resource availability.
- At 18:16:27 UTC, Slurm reported 3B RUNNING on deep-chungus-7 and 7B PENDING for Dependency.

## Checkpoint continuity

The 3B log previously reached optimizer update 749, but the latest paired trainer/orchestrator checkpoint is step 725. Both trainer shard files, trainer metadata and orchestrator progress exist and are nonempty. The deployed manifest SHA-256 and the run's recorded manifest both equal `0286e765ba0f9d8f30816b9144175557106055430bd526c6bec5ee3f3218ea99`. No experiment settings or tracked files changed.

For 7B, the latest verified paired checkpoint before pausing is step 150. Step 175 has shard files and orchestrator progress but lacks trainer metadata, so it was not counted as a valid resume checkpoint. The existing resume wrapper preserves incomplete checkpoints in recovery storage.

## Evidence

Local audit files are under `tmp/resume-3b-stale8-chungus7-20260918/`. The remote scheduling receipt is `/mnt/nfs/home/mohamadzbib/projects/rl-infra/diagnostics/resume-3b-stale8-chungus7-20260918/priority-switch.json`. The remote experiment retains `relocation-deep-chungus-7-20260918.json` separately from the original submission receipt.

At 18:19:48 UTC, the batch health probe passed with world size 4, BF16 backward true and NCCL sum 10. All four allocated devices reported A100 80GB. Model/data/tokenizer/custom-loss validation and the existing smoke gate also passed. Model staging and checkpoint loading were still in progress at that check.

## Follow-up at 19:01 UTC

Job 2141858 resumed from step 725, with CUDA_VISIBLE_DEVICES and SLURM_JOB_GPUS both 0,1,2,3 and PRIME_VISIBLE_DEVICES 2,3,0,1. The current attempt completed optimizer updates through 749 and started saving step 750 at 18:57:35 UTC. Step 750 now has trainer metadata, both trainer shards and orchestrator progress. No fatal errors were found in the recent trainer, orchestrator or inference log tails. Job 2141895 remains pending on the configured dependency. Evidence: `tmp/cluster-health-20260918/check.raw.json`.
