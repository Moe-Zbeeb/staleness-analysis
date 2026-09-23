# 7B staleness ≤6 GPU mapping recovery

At **2026-09-20 01:20:59 UTC**, the user explicitly requested node9 placement. Existing job **2142067** was held, pinned to **deep-chungus-9**, changed to partition **low-priority**, and released at **normal QoS**, preserving six GPUs, 64 CPUs, 384 GiB and checkpoint125. It was PENDING immediately after release. Node9 had five schedulable GPUs free at preflight. See the [placement receipt](placements/job-2142067-node9-20260920.json). The earlier unrestricted placement below is historical.

## Original resubmission

At **2026-09-20 01:17:23 UTC**, job **2142067** was **PENDING**, reason **Priority**, at normal priority. It requests six GPUs on one compatible node, 64 CPUs and 384 GiB host memory, resuming the paired checkpoint at step125. It is released and has no dependency or node pin. The user previously authorized normal-priority scheduling on any compatible node with six available GPUs.

The previous job2141991 failed on deep-chungus-10 with a GPU UUID mismatch before training. This package derives the allocated device files from the reviewed Slurm GRES configuration, maps their driver UUIDs to canonical CUDA ordinals, and rejects GPUs with existing compute processes or insufficient memory. The frozen six-GPU launcher is changed only to verify this package and select those canonical devices before its existing six-rank GPU health test.

Four trainer GPUs and two inference GPUs are retained. The original six-GPU configuration overlay, runtime, optimizer, data, rewards, staleness, checkpoint paths, recovery capture, run lock, requeue handling and Comet project remain unchanged. The original frozen package is still used for experiment gates, checkpoint capture and completion audits.

## Validation

- Eight local CPU tests pass, covering noncontiguous allocations, exclusion of another job's devices, two-inference/four-trainer assignment, hardware class, occupied devices, insufficient memory and the exact launcher delta.
- The shell syntax check passes.
- Remote package verification confirmed every new file hash, every original recovery file hash, the exact two launcher changes, and the prior passing CPU configuration validation from job2141990.
- Checkpoint125 metadata/progress hashes and all recorded file sizes match the original evidence. No newer paired checkpoint was present.
- Live GPU startup, BF16 backward, NCCL sum21, full-state restoration and subsequent training updates remain unverified while queued.

Eligible nodes remain deep-chungus-[7-11] and deep-h-[1-3]. Down or drained nodes cannot currently dispatch. There was no healthy compatible node with six free GPUs at submission; dispatch also depends on queue order.

## Evidence

- [Exact submission and checkpoint preflight](training-submission.json)
- [Latest Slurm queue verification](queue-verification.json)
- [Remote package verification](deployment-verification.json)
- [Exact launcher changes and source provenance](provenance.json)

New package manifest SHA-256: `c15cb8c5a84b8fd02e66e67b998e85fdd0d65d5933ab397b56a8bff3e25f203d`.

Frozen six-GPU package SHA-256: `81382a61bb3d8d1684bf686dc51b2845377190aee3042b07a505022b00dd6ee1`.

Remote package: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/recoveries/7b-stale6-device-mapping-20260920`.

## Superseded by five-GPU request

The user subsequently requested five GPUs on deep-chungus-9. Six-GPU job2142067 was cancelled after a verified five-GPU submission. The current queued continuation is job2142073, normal priority, four trainer ranks plus one inference rank, waiting for conflicting job2142035 to terminate. See the [five-GPU recovery record](../7b-stale6-five-gpu-20260920/README.md).
