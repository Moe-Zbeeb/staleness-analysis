# 7B staleness ≤4 restart on deep-chungus-7

Job **2142066** was submitted at **2026-09-20 00:44:43 UTC**, using normal priority and five A100 GPUs on `deep-chungus-7`. The user requested restarting the dropped run and had authorized placement elsewhere. The existing four-trainer/one-inference topology and paired checkpoint225 remain unchanged.

## Why the node9 attempt stopped

Job2142052, restart1, failed with exit1:0 after6m38s. Its prelaunch GPU check found PID3248566 using10,012MiB on allocated UUID `GPU-012e7790-9bf9-122d-b030-f88e489a373e`. The check stopped before model staging or checkpoint restoration. The process owner and reason it occupied a Slurm-assigned device were not established; no process was signaled. This failure is separate from the monitoring SSH connection closing.

The selected node7 had five schedulable GPUs, 126 free CPU slots and sufficient Slurm-accounted memory. Its other allocation, job2136036, held three GPUs. The launch derives and checks its actual assigned device identities. The startup gate remains mandatory.

## Verification

At **01:01:36 UTC**, the job was actively restoring checkpoint225. A read-only diagnostic within its allocation found each of the four trainer processes holding its corresponding `__0_0.distcp` through `__3_0.distcp` file open. Every process's read counters increased over five seconds. The inference server was healthy and the orchestrator had restored step225. Full trainer restore completion, the initial weight transfer and new training steps remained unverified.

At **00:49:40 UTC**, Slurm reported RUNNING, restart0, on node7. Runtime, experiment and GPU checks passed. All five assigned GPUs were empty before launch; BF16 backward passed and NCCL returned the expected sum15 at world size5. Canonical CUDA ordinals were `3,2,1,0,5`, with PRIME role order `5,3,2,1,0`. Full checkpoint restoration and subsequent training updates remained pending while staging model files.

- [Exact submission and Slurm verification](training-submission.json)
- [Failure logs and accounting](failed-attempt-inspection.json)
- [Candidate capacity](capacity.json)
- [Passing GPU health and current-attempt runtime checks](startup-health.json)
- [Latest startup and component logs](latest-startup.json)
- [Checkpoint read progress across all four trainers](checkpoint-restore-io.json)
- [Read-only memory and process snapshot](node-pressure.json)

Frozen package SHA-256: `0bd3e9f6d022ff2ab1a0c0126deedb57388434f2da28eb28633dec59a438ef5b`.

Remote submission receipt: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/recoveries/7b-stale4-weight-sync-20260919/resubmissions/node7-20260920/training-submission.json`.
