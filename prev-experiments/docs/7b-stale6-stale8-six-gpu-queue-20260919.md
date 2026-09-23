# Six-GPU scheduling for 7B staleness caps 6 and 8

## September 20 update

At **12:04:14 UTC**, cap6 replacement **2142129** started on all **eight A100 GPUs of deep-chungus-9**, **normal priority**, resuming checkpoint **125**, with **Comet disabled at the user’s request**. The original four-trainer/four-inference topology is restored. The startup weight wait is extended to 7,200 seconds after job2142075 failed its 1,200-second startup-broadcast wait. Local metrics and all checkpoints/evaluations remain enabled. Eight-rank GPU identity/BF16/NCCL health passed at the 12:08 UTC check (sum36), and PRIME startup was underway. See the [full-node launch record](../recoveries/7b-stale6-full-local-20260920/README.md) for validation and current startup evidence.

At **02:10 UTC**, cap8 replacement **2142080** was queued on **deep-chungus-11**, **normal priority**, with **five A100 GPUs (four trainer + one inference)**, checkpoint **100**, and dependency `afterany:2142020`. The node's five free GPUs were confirmed, but only about46–50 GiB host RAM was available. Read-only diagnostics traced the dominant data-worker memory usage to job2142020. The continuation requests384 GiB and checks available host memory again before startup. Superseded six-GPU job2141992 was cancelled; all other training/evaluation jobs were preserved. Native Slurm CUDA assignment is retained. See the [cap8 submission and memory evidence](../recoveries/7b-stale8-native-cvd-20260920/README.md). No cap8 training has started on node11 at this snapshot.

At **01:57:18 UTC**, cap6 job **2142075** was RUNNING on node9 with five GPUs at normal priority. Native GPU assignment, occupancy and five-rank BF16/NCCL checks passed; the collective sum was15. Model staging was underway for checkpoint125. Full restore and new training steps were not yet verified. See the [startup evidence](../recoveries/7b-stale6-native-cvd-20260920/startup-2142075.json).

At **01:53 UTC**, cap6 replacement **2142075** was submitted for **five GPUs on deep-chungus-9**, **normal priority**, checkpoint125, with **no dependency**. Probe2142074 completed successfully and showed that Slurm's original five visible GPUs were all idle. Our prior extra mapping from CUDA selectors to device-file minors selected an occupied GPU incorrectly. Thus the previous occupancy diagnosis did not apply to Slurm's original allocation; the dependency on2142035 was unnecessary. Superseded job2142073 was cancelled. The [corrected launcher](../recoveries/7b-stale6-native-cvd-20260920/README.md) preserves Slurm's CUDA-visible list and validates the five actual UUIDs before training. Other jobs were unchanged.

At **01:38 UTC**, the user's later request to use the five GPUs already available on node9 superseded cap6's six-GPU request. Cap6 is now job **2142073**, **five A100 GPUs (four trainer + one inference)**, **normal priority**, pinned to **deep-chungus-9**, resuming **checkpoint125**, with dependency `afterany:2142035`. Five-GPU config/checkpoint validation passed in the preceding attempt2142069, but startup stopped because frankzydou's process3248566 was using allocated physical GPU minor4. Read-only diagnostics confirmed that process belongs to running job2142035. The new job waits for that job to end and will repeat the occupancy/health checks. Original job2142067 was cancelled; attempt2142069 failed before training. See the [five-GPU recovery package and evidence](../recoveries/7b-stale6-five-gpu-20260920/README.md). Cap8 and all other training/evaluation jobs were left unchanged.

At **01:20:59 UTC**, the user requested pinning cap6 directly to **deep-chungus-9**. Job **2142067** was pinned there and released with partition **low-priority**, QoS **normal**, and its existing six-GPU topology. The job was PENDING after release; node9 had five GPUs free at preflight. This supersedes the earlier unrestricted placement for cap6. See its [placement receipt](../recoveries/7b-stale6-device-mapping-20260920/placements/job-2142067-node9-20260920.json).

At **01:17:23 UTC**, cap6 replacement job **2142067** was queued at **normal priority**, reason **Priority**, to resume checkpoint125 on one compatible six-GPU node. The previous job2141991 failed its GPU UUID check. The [separate mapping correction](../recoveries/7b-stale6-device-mapping-20260920/README.md) preserves the validated four-trainer/two-inference setup and frozen experiment. Eight CPU tests and remote package verification passed; GPU startup remains pending. Existing cap8 job2141992 remains queued for Resources and was unchanged by this cap6 recovery.

## Original September 19 submission

Verified 2026-09-19 at 02:34:18 UTC. The user requested both stopped H100 runs queue on any compatible node with six free GPUs, retaining the earlier normal-priority choice.

| Run | New job | Resume checkpoint | State | Pending reason |
| --- | --- | --- | --- | --- |
| Qwen2.5-Math-7B, staleness ≤6 | 2141991 | 125 | PENDING | Resources |
| Qwen2.5-Math-7B, staleness ≤8 | 2141992 | 100 | PENDING | Priority |

Both jobs use account `grad-students`, partitions `low-priority,h100`, QoS `normal`, one node, six GPUs, 64 CPUs and 384 GiB host memory. Neither is held or pinned to a particular node. The eligible node set is `deep-chungus-[7-11]` and `deep-h-[1-3]`; Slurm will only dispatch to available nodes. The continuation profile requires 80GB-class A100/H100 GPUs; older 40GB A100, V100 and 2080Ti nodes are excluded. Each job must fit on one node with enough GPU, CPU and memory resources, subject to queue priority.

Superseded h3-only job **2141971 was cancelled** after both replacement submissions were created and checked. The other running jobs were preserved: 2141453 (14B), 2141895 (7B ≤4), and 2141859 (3B ≤6).

## Continuation and provenance

Trainer world size remains four, matching the four distributed checkpoint shards and preserving model, optimizer, scheduler and trainer progress restoration. Inference changes from four replicas to two at tensor parallel size one. The model, reward, data, tokenizer, training settings, staleness caps and evaluation schedule remain the original ones. Existing run/observer directories, model links and Comet projects are preserved. The original names still contain `8h100`; this document and the recovery specification identify the new execution segment.

Frozen source and data manifests were not changed. The recovery package is in `staleness-analysis/recoveries/7b-six-gpu-20260919` locally and `/mnt/nfs/home/mohamadzbib/projects/rl-infra/recoveries/7b-six-gpu-20260919` remotely. It contains the configuration overlay, launchers, allocation health probe, exact submission receipts and checkpoint evidence. Package-manifest SHA256: `81382a61bb3d8d1684bf686dc51b2845377190aee3042b07a505022b00dd6ee1`.

The recovery guard validates the original smoke/source evidence and exact overridden configuration, captures checkpoint evidence before every resumed attempt, and retains evidence for ordinary checkpoint pruning. Final completion reports will be saved under the recovery package's `validation/` directory. Future publication or completion checks must account for this overlay rather than comparing the resolved inference topology directly with the frozen eight-GPU config.

## Validation and remaining startup checks

CPU validation job **2141990 completed 0:0**. Both original smoke gates passed. The only resolved configuration differences were allocation 8→6, inference replicas/API workers 4→2, and the corresponding broadcast inference worker counts. Distributed checkpoint progress was 125 and 100, with optimizer metadata present; orchestrator next-step counters were 126 and 101. Noncontiguous allocation mapping was verified using `0,2,3,4,6,7` → `6,7,0,2,3,4`.

An earlier preparation-only validation job, 2141989, stopped on an incorrect assertion equating the periodic orchestrator next-step counter with the completed trainer step. The assertion was corrected after checking PrimeRL's save/resume implementation. Its logs, original validation script and package manifest were preserved under the remote recovery package's `validation-history/2141989`; no training was launched by that job.

Actual GPU startup has not happened at this snapshot. Each training allocation must pass a bounded six-rank UUID/BF16/NCCL check before model staging, then reload its saved checkpoint. Slurm PENDING confirms scheduling only. Hardware and inference-throughput changes are recorded comparison factors for later staleness analysis.
