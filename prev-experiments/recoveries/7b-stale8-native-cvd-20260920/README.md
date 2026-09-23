# 7B staleness ≤8 continuation on deep-chungus-11

Job **2142080** was submitted and released on September 20 at **02:10 UTC**, pinned to **deep-chungus-11**, with account `grad-students`, partition `low-priority`, QoS `normal`, five A100 GPUs, 48 CPUs and 384 GiB host memory. It is **PENDING**, with dependency `afterany:2142020`, and has no GPUs allocated yet. The superseded six-GPU job **2141992** was cancelled after the replacement was submitted and verified.

The continuation resumes **checkpoint 100**, retaining four trainer ranks and using one inference replica. It preserves the original cap8 experiment, source/data manifests, optimizer/scheduler state, training hyperparameters, evaluation schedule, output/observer directories and Comet project. Original run names retain their historical `8h100` suffix.

## Why immediate startup is blocked

The node has eight A100 GPUs, of which three are allocated to job 2142020 and five are schedulable. Host memory is the limiting resource. CPU-only diagnostics 2142077–2142079 measured approximately 46–50 GiB of `MemAvailable`, with about 900 GiB in anonymous memory. Most resident memory belonged to frankzydou's data workers. A process-parent trace confirmed the largest worker belongs to `slurmstepd: [2142020.batch]`.

The initial memory preflight deliberately exited with failure because available RAM was below the continuation's 384 GiB request. This was a diagnostic failure, not a failed training attempt. Reclaimable cache was insufficient to account for the shortage; ZFS ARC was already near its minimum. Slurm reported `AllocMem=0`, so its resource summary did not capture the actual host-memory pressure.

The replacement waits for job 2142020 to terminate. A new host-memory check runs inside the eventual allocation and requires 384 GiB available before checkpoint validation/model startup. GPU occupancy, identity, BF16 and NCCL checks also remain mandatory. No other user's job or process was changed.

## Validation and pending checks

Checkpoint 100 has all four nonempty distributed trainer shards plus matching trainer metadata and orchestrator progress. Source, data, guard and checkpoint metadata/progress hashes matched the frozen cap8 evidence. Nine native-GPU mapping/occupancy regression tests passed locally, as did Python compilation and shell syntax. All 17 deployed package files matched the final manifest.

The corrected launcher preserves the original Slurm `CUDA_VISIBLE_DEVICES` and verifies its physical UUIDs rather than substituting device-file minor numbers. The included node9 probe is a regression fixture only; actual node11 GPU health has not been checked by this queued training job.

Full five-GPU resolved-config/checkpoint validation, GPU startup, checkpoint restoration and new training steps remain pending until the dependency clears and Slurm starts the job. The final package hash supersedes the initial pre-memory-gate hash; the initial deployed files are preserved remotely under `pre-memory-gate-history`.

## Evidence

- [Submission, exact command and verified Slurm settings](training-submission.json)
- [Node, queue and checkpoint preflight](preflight.json)
- [Host-memory diagnostic](memory_detail.json)
- [Memory-consuming process's job attribution](attribute_memory.json)
- [Package manifest](package-manifest.json)
- [Derivation and frozen inputs](provenance.json)

Final package-manifest SHA256: `2f247e46bf1fdd5032a8507c5976c2cd6269a7cc180e4f38677c374b4463ea5f`.

Remote package: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/recoveries/7b-stale8-native-cvd-20260920`.
