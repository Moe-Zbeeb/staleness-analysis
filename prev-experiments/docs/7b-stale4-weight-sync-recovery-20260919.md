# 7B staleness ≤4 recovery, 2026-09-19

## Current state

At **2026-09-20 01:01:36 UTC**, job **2142066** was actively restoring checkpoint225 on **deep-chungus-7**, with five A100 GPUs at normal priority. All four trainer processes had their checkpoint shard open and increasing read counters. The inference server was healthy and the orchestrator had restored step225. Full trainer restoration, initial weight transfer and subsequent training steps remained unverified. Evidence is in the [node7 restart receipt](../recoveries/7b-stale4-weight-sync-20260919/resubmissions/node7-20260920/README.md).

## Node7 startup checks

At **2026-09-20 00:49:40 UTC**, replacement job **2142066** was **RUNNING on deep-chungus-7**, normal priority, five A100 GPUs. All five allocated GPUs were empty and passed BF16 backward and the five-rank NCCL sum15 test. Runtime and experiment checks passed. Model staging was underway; restoration of checkpoint225 and new training steps remained unverified. See the [node7 restart receipt](../recoveries/7b-stale4-weight-sync-20260919/resubmissions/node7-20260920/README.md).

The node9 attempt of job2142052 failed with exit1:0: the GPU occupancy check found PID3248566 using10,012MiB on an allocated GPU. It stopped before loading the checkpoint. The monitoring SSH disconnect was a separate event.

## Earlier node9 attempt

At **23:20:39 UTC**, job **2142052**, restart **1**, was **RUNNING on deep-chungus-9** after the user's authorization to relocate it. Normal priority, five GPUs, four trainer ranks and checkpoint225 were preserved. The current attempt passed isolated-runtime verification; GPU health, checkpoint restoration and new training steps were not yet verified. The next read-only SSH connection closed before producing a report, so automated connection attempts stopped. See the [dated relocation receipt](../recoveries/7b-stale4-weight-sync-20260919/relocations/job-2142052-node9-20260920/README.md).

## Earlier node10 attempt

At **18:20 UTC**, recovery job **2142052** is running on **deep-chungus-10**, using **normal priority and five A100 GPUs**. It has passed the isolated-runtime, experiment and GPU health checks. Model staging, checkpoint restoration and consecutive successful training updates are still pending; these startup checks do not establish that training has resumed.

| Item | Verified value |
| --- | --- |
| Previous failed job | 2142033 |
| Restart job | 2142052 |
| Account / partition / QoS | grad-students / low-priority / normal |
| Layout | Four trainer GPUs, one inference GPU |
| Resume checkpoint | 225, with optimizer, scheduler and progress state |
| CPU cluster validation | 2142051, COMPLETED, exit 0:0 |
| GPU startup | Five assigned GPUs empty; BF16 backward passed; NCCL sum 15 at world size 5 |
| Local tests | 32 passed; both shell syntax checks passed |
| Shared Prime revision | ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1 |

The fresh SSH attempt succeeded through the jump host after Duo approval. Checkpoint 225 is the latest complete paired checkpoint. Step 239 contains only orchestrator progress; the launcher will preserve it separately before resuming.

## Recovery change

The [isolated package](../recoveries/7b-stale4-weight-sync-20260919/README.md) changes only the job-local weight-update client. It sends each pause, update and resume request once, uses a 3,600-second read timeout, and stops on an uncertain update without resuming inference or advancing its policy. Tests reproduce the original duplicate-request behavior and verify the captured receiver/watcher sequence with the replacement.

The cluster validation loaded checkpoint progress and verified trainer step 225, orchestrator next step 226, four trainer shards, and model/optimizer/scheduler/progress coverage. The resolved experiment differs only by the already-established startup wait of 7,200 seconds. Training data, rewards, model, staleness, optimizer, sampling and GPU topology remain frozen.

The shared repository is unchanged. Both cluster validation and training startup verified all 238 original source/configuration files and confirmed that module imports resolve to the isolated runtime. The GPU gate verified that all five assigned GPUs had no pre-existing compute processes. Slurm minors 3–7 mapped to CUDA ordinals `0,7,6,5,4`; PRIME uses `4,0,7,6,5` to place inference first. The five-rank health test passed BF16 backward and the expected NCCL sum of 15. No device outside the allocation was selected.

## Shared-storage observation

A read-only CPU snapshot inside job 2142052 at 18:12 UTC found approximately **755 GiB of available RAM**, no recent memory pressure, and substantial I/O pressure. Fifty-four of this user's processes were waiting in `rpc_wait_bit_killable`. The shared NFS filesystem was 98% used with approximately 3.12 TiB free, and the node-local filesystem had approximately 501 GiB free.

These measurements support current RPC/storage waiting as a bottleneck; they do not prove the cause of the earlier weight-transfer timeout. No host configuration, storage mount, driver or other user's process was changed.

## Evidence

- [Fresh checkpoint and job inspection](../../tmp/recover-7b4-weight-sync-20260919/reconnect_preflight.raw.json)
- [Passing cluster validation](../../tmp/recover-7b4-weight-sync-20260919/validation_resources.raw.json)
- [Exact training submission and verified Slurm fields](../../tmp/recover-7b4-weight-sync-20260919/submit_training.raw.json)
- [Latest startup checks](../../tmp/recover-7b4-weight-sync-20260919/check_recovery_9.raw.json)
- [Host pressure snapshot](../../tmp/recover-7b4-weight-sync-20260919/node10-pressure-summary.json)

Package manifest SHA-256: `0bd3e9f6d022ff2ab1a0c0126deedb57388434f2da28eb28633dec59a438ef5b`.

Runtime provenance SHA-256: `969f5e9dca7f205ba53c92dce124fef8efbb04832a02eda37977faaacaebfa22`.
