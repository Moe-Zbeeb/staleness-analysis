# Qwen 3B cap 6 and 7B cap 4 normal-priority recovery

## Request and placement

The user requested recovery of the failed 3B cap-6 run alongside the 7B cap-4 run, both at normal priority. The user explicitly chose to keep both on `deep-chungus-10`.

The planned allocations preserve the latest approved topology: 3B uses two trainer GPUs and one inference GPU; 7B uses four trainer GPUs and one inference GPU. Together they request all eight A100 GPUs without overlapping allocations. Both jobs use account `grad-students`, partition `low-priority`, and QoS `normal`.

## Failure and remedy

3B job `2142027` failed after the orchestrator's startup weight wait expired at 1,200 seconds. Its trainer was restoring checkpoint 775 from NFS; an earlier read-only process check confirmed that checkpoint read counters were increasing. The orchestrator raised `TimeoutError` while waiting for the trainer's startup broadcast. The launcher then terminated its processes.

7B job `2142029` was subsequently preempted and requeued. Bazzi's eight-GPU background allocation restarted on chungus-10. Live Slurm configuration uses partition-priority preemption; the background partition has priority tier 5 and the normal jobs' low-priority partition has tier 1.

New immutable recovery packages set only `orchestrator.ckpt.wait_for_weights_timeout = 7200` relative to the preceding recovery configuration. The original configuration field is unset, and the pinned framework uses a 1,200-second fallback. This change gives checkpoint loading up to two hours to publish startup weights; it does not make NFS faster or establish that the next startup has succeeded.

The original experiment source and data manifests remain unchanged. Checkpoints, optimizer, scheduler, seed, sampling, token limits, staleness bounds, output paths, and Comet identity are preserved. Full-state resumes use checkpoint 775 for 3B and checkpoint 225 for 7B. No checkpoint is converted to a model-only restart.

## Validation and provenance

- 3B recovery: `recoveries/3b-stale6-normal-retry-20260919`, manifest SHA-256 `de0758505a26381636fb1834c8fa5c675802db51c9e836fe746c27d9e1c45029`.
- 7B recovery: `recoveries/7b-stale4-normal-retry-20260919`, manifest SHA-256 `9e766f83ec62805dcddd55b66b202f09041a8a54eef38a751275d052e8eb0d2f`.
- Local validation passed 20 3B tests and 13 7B tests, including device mapping, retained checkpoint provenance, overlay binding, and runtime audit adaptation. Python and shell syntax checks passed.
- CPU-only cluster preflights: `2142030` for 3B and `2142031` for 7B, on `deep-chungus-6` at normal priority. Both completed with exit `0:0`. The saved trainer steps were 775 and 225, and the orchestrator next steps were 776 and 226. Their checkpoint metadata included model, optimizer, scheduler and progress state. The exact resolved configuration comparisons passed.
- Local operation records: `tmp/normal-retry-chungus10-20260919/` in the workspace root.
- Remote immutable packages and mutable validation/submission receipts: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/recoveries/` under the package names above.

Submission requires each cluster preflight to complete successfully, verifies the paired checkpoint identity and exact configuration changes, creates replacements held, checks their Slurm settings, cancels only the old held pending 7B job, and releases the replacements. Live GPU health and training progress must be checked after allocation; a queued job is not a completed startup.

## Submitted jobs

Submitted and verified at approximately 2026-09-19 12:35 UTC:

| Run | Job | GPUs | Priority | Requested node | Resume checkpoint | Status |
| --- | --- | ---: | --- | --- | ---: | --- |
| Qwen2.5-3B cap 6 | 2142032 | 3 | Normal | deep-chungus-10 | 775 | Pending, Priority |
| Qwen2.5-Math-7B cap 4 | 2142033 | 5 | Normal | deep-chungus-10 | 225 | Pending, Priority |

Both replacements were released and are not held. Old pending 7B job `2142029` was cancelled only after both replacements had been created and verified while held. Neither active training nor another user's allocation was cancelled. Bazzi's job `2140333` still occupies all eight GPUs on chungus-10. The 14B job `2141453` remains running on chungus-3. No new GPU startup or training step has occurred for these two retries yet.

The exact commands, verified Slurm fields, validation hashes and node snapshot are in `tmp/normal-retry-chungus10-20260919/submit_training.raw.json` and each remote recovery package's `training-submission.json`.

## Later user override: 3B high priority

The user subsequently explicitly requested promoting the 3B job to high priority on chungus-10, allowing Slurm to preempt its eight-GPU background allocation and place the normal-priority 7B job on the remaining five GPUs.

Job `2142032` was updated in place with `scontrol update JobId=2142032 Partition=high-priority QOS=high-priority JobName=qwen3b-stale6-3gpu-hp`. The three-GPU request, checkpoint, recovery package and training configuration are unchanged. Its immutable package still records the original normal-priority request; the newer authorization and effective Slurm fields are recorded separately in the remote package's `priority-change-high-priority.json` and local `tmp/normal-retry-chungus10-20260919/promote_3b_hp.raw.json`.

At 2026-09-19 12:41:20 UTC, both `2142032` (3B, high priority, three GPUs) and `2142033` (7B, normal priority, five GPUs) were `RUNNING` on chungus-10. Both initial configuration gates passed. GPU health checks and saved-state loading were still pending at that observation. No other user's job was manually cancelled; Slurm handled allocation and preemption.

At 12:43:43 UTC, both launchers had completed. Three-rank and five-rank GPU probes passed finite BF16 backward and NCCL collective sums of 6 and 15. The allocation UUIDs were disjoint and covered all eight GPUs. Actual framework assignments were 3B inference CUDA ordinal 1 with trainer ordinals 3 and 2; 7B inference ordinal 4 with trainer ordinals 0, 7, 6 and 5. These ordinals differ from Slurm device minors; the verified UUID mapping remains authoritative.

Both current resolved orchestrator configurations set the 7,200-second startup wait. Trainer and orchestrator resume steps are 775 for 3B and 225 for 7B, with optimizer, scheduler and progress restore enabled. Both Comet bridges connected. Saved-state startup was still in progress; no new training step was observed. Evidence is saved in `hp-gpu-verification.json`, `hp-launch-verification.json` and timestamped `startup-*.json` under the local operation directory.
