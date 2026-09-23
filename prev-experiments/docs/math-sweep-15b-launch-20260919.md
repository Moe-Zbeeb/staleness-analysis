# First math sweep: completed 1.5B family

## September 20 continuation on node11

At **12:27 UTC**, replacement **2142130** was **RUNNING on two free A10080GB GPUs of deep-chungus-11 at normal priority**. The corrected wrapper preserves native Slurm CUDA selectors and validates their UUIDs directly; both assigned GPUs passed occupancy/memory checks. The original 80 node11 cells are resumed with frozen evaluation settings. Model startup tests and new production generation were still pending at this snapshot. Completed node1 job 2142048 retains 60 cells/15,175 answers.

Existing report **2142054** now depends on `afterok:2142130`; its failed dependency was replaced without submitting a duplicate report. All 140 cells must pass the unchanged final verification. See the [native GPU continuation](../evaluation/launches/15b-math-normal-v2-native-20260920/README.md) and its submission receipt before any later retry. This supersedes the stopped state in the earlier September 20 status below.

## September 20 status and corrected failure diagnosis

At **12:15 UTC**, node1 worker **2142048** is **COMPLETED, exit 0:0**, with all **60 assigned production cells** complete and **15,175 saved answers**. These are receipt counts; this status check did not rehash all answer records. Node11 worker **2142053** failed before production generation, leaving **80 cells** unfinished. Report **2142054** is pending with `DependencyNeverSatisfied`. No evaluation GPU worker is currently running or queued.

The node11 failure receipt recorded original Slurm CUDA selectors `3,4`. The frozen evaluation mapper still incorrectly treats those selectors as Linux NVIDIA device-file minor numbers, selecting UUIDs `GPU-a2515c5d-031e-cb1f-3e57-d6810637038c` and `GPU-8700e8c0-2ab7-4c31-3e0a-bcfc35597de8`. The occupancy guard then rejected PID2649442 on the first UUID, using9,532 MiB; the recorded process path belongs to frankzydou. The same node's nearby native-CUDA health probe instead resolved CUDA selectors3 and4 to UUIDs `GPU-756ddcc4-fe19-51ef-baa2-6f633b89829c` and `GPU-8ceb7e35-48da-9616-c892-129f3109918c`. The GPU identity mistake is in our launcher; the guard correctly prevented generation on the device it was given. This does not establish that Slurm's original two visible GPUs were occupied.

Before resubmission, replace the minor-number translation with validation of the unchanged Slurm CUDA-visible assignment, retaining occupancy checks, A10080GB comparison grouping, frozen evaluation settings and all completed cells. Update the dependent report job to the replacement worker. This status inspection did not submit or cancel jobs. The later 7B full-node recovery uses a corrected mapper, but that does not update this frozen evaluation launcher.

Evidence: [saved evaluation status and failure allocation](../evaluation/launches/15b-math-normal-v2-occupancy-20260919/status-20260920.json). Earlier snapshots below are historical.

**Current launch, verified September 19 at 18:14 UTC:** normal-priority job **2142048** continues evaluating on one GPU on deep-chungus-1. Replacement **2142053** is pending for two GPUs on deep-chungus-11, after both existing node11 jobs **2142020/2142041** end. CPU report **2142054** waits for successful completion of **2142048 and 2142053**. Conflicted node11 evaluation **2142049** and obsolete report **2142050** were cancelled; all preparations, results and logs are preserved. No combined benchmark scores are ready. Earlier job IDs below are retained as history.

On September 19 the user requested starting evaluation with Qwen2.5-Math-1.5B because its base and all four staleness versions are ready. This authorizes this family to run before the other model families finish.

## Scope

Five immutable model revisions: starting model and final step-1000 staleness caps 2, 4, 6 and 8. The frozen default native-context configuration covers 12 benchmarks, 140 cells and 48,620 responses. Experimental context extension is outside this launch.

The full data audit contains 1,740 questions. One parameterized MATH500 question is conservatively excluded from the separate training-overlap-filtered score, leaving 1,739. A second near match was reviewed and retained because its mathematical objective and constraints differ. There are no unresolved candidates or cross-benchmark duplicates. Raw source answers are preserved.

## Original submitted jobs

| Job | Role | Initial verified state | Resources |
| --- | --- | --- | --- |
| 2142039 | CPU model/data preparation | Running on deep-chungus-6 | 4 CPUs, 16 GiB requested, 2 hours |
| 2142040 | GPU smoke then full evaluation | Pending, afterok:2142039 | One exclusive 8-A100 node, 32 CPUs requested, 192 GiB requested, 24 hours |

Both jobs explicitly use account `grad-students`, partition `high-priority` and QoS `high-priority`. The GPU candidate set is deep-chungus-7, deep-chungus-9 or deep-chungus-11, each configured with eight GPUs. The other nodes are excluded through Slurm's exclusion list; no multi-node nodelist is requested. Existing training on deep-chungus-3 and deep-chungus-10 is preserved. At submission the user's 12-GPU high-priority allowance was fully occupied, so successful preparation alone does not make the evaluation immediately runnable.

At 14:25:26 UTC, CPU preparation was progressing with three of 140 cells indexed and no error. This verifies that the production runtime, pinned base snapshot, frozen tokenizer and grader passed preparation for those first cells. The GPU job remained pending on the preparation dependency; no GPU smoke or benchmark generation is claimed yet.

Exact commands and verified Slurm fields are in [the submission receipt](math-sweep-15b-submission-20260919.json). A queued/running Slurm state does not establish GPU validation or benchmark completion.

## Immutable deployment and results

- Source: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/evaluation-runs/15b-math-20260919/source`
- Output: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/outputs/math-sweep-15b-20260919`
- Source manifest SHA256: `c7794f9212c5bfc957f8c94e3a95af231371ddb1a31e62033db1b73b03ef4ebe`
- Package archive SHA256: `9783602a08d262de0ab967584bdebca3136015664b0f7ce3d67a578e6b9c19bb`
- Source plan SHA256: `e1a9974cbf8074c4eea8dfc71cd12018bc51c8a7982cfa7310392f0bfc3ea44c`
- Data bundle SHA256: `7b419e0af7d830bcaaa147741e136cff8d80716311684c6bc94ad1e47e789eaf`

The output contains the source plan, data bundle/receipt, overlap decisions, preparation index, job submission receipt and logs. Preparation writes a final identity for each cell. The GPU dispatcher maps the complete physical node to distinct GPU UUIDs, runs an isolated smoke generation on every GPU, and starts the production queue only after all smoke checks succeed. Every GPU receives one worker. Raw answers, token IDs, scoring fields and completion receipts are durable. Smoke outputs are kept outside production result directories.

Local validation passed 74 evaluation tests, six launcher tests and both shell syntax checks. The five published releases' metadata, step, staleness, original base identity, training-data hash and weight-shard coverage were independently checked. Allocated CPU preparation still verifies actual model weights and runtime versions; actual GPU validation is the initial dependent job phase.

Do not modify the deployed source after preparation or resubmit this matrix without inspecting existing receipts. Later all-family sweeps must account for these cells and avoid duplicate evaluation. For the original launch, `prepared/preparation-index.json` and `results/.dispatch/` retain its evidence; monitor the current jobs and v2 result paths listed below for active progress. No prior AIME 8K results fill cells in this launch.

## First normal-priority attempt on September 19

The user subsequently authorized normal priority on two available GPUs each of deep-chungus-1 and deep-chungus-11. Original CPU preparation completed all 140 cells. Its source, data, model snapshots and preparation receipts remain immutable. The replacement package is [15b-math-normal-20260919](../evaluation/launches/15b-math-normal-20260919/README.md).

Chungus-1 uses 40GB A100s and chungus-11 uses 80GB A100s. Each complete benchmark/profile comparison group—including all five model versions and every token budget—stays on one node. The launcher derives separate hardware-bound cell identities without changing prompt bytes, models, tokenizer, decoding, seeds, grading or engine configuration. Node 1 receives 60 cells / 24,080 responses; node 11 receives 80 cells / 24,540 responses. Maximum output-token work is nearly equal across nodes.

Each partial allocation requests exactly two GPUs, eight CPUs and 24 GiB host memory, without exclusivity. Slurm GPU IDs are resolved through a reviewed GRES configuration to device minors and UUIDs; startup checks the actual GPU class and available memory. Both nodes smoke-test all five models before beginning their disjoint production cells. Twenty new launcher tests and shell syntax checks passed. A combined report requires complete, verified receipts for all 140 cells.

- Replacement source: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/evaluation-runs/15b-math-normal-20260919/source`
- Replacement output: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/outputs/math-sweep-15b-normal-20260919`
- Replacement launcher manifest SHA256: `fe248130b55ec695928d0b15714ffb2ef4ab5305a0cdb415a5d308f1b7506f25`
- Replacement archive SHA256: `0f5aef28e68b2ffac95cc271ddbb5653714c617bb561afb01c0ff74b66efbb0d`

Original pending GPU job **2142040 was cancelled** before releasing the replacement jobs. The original full-matrix lock also excludes simultaneous dispatch, and per-shard locks prevent duplicate replacement workers. Training allocations are outside this replacement.

### First replacement jobs (failed before generation)

| Job | Role | Node | Resources | Verified initial state |
| --- | --- | --- | --- | --- |
| 2142044 | Evaluation shard 0 | deep-chungus-1 | 2 A100 40GB GPUs, 8 CPUs, 24 GiB requested | Running; startup verification |
| 2142045 | Evaluation shard 1 | deep-chungus-11 | 2 A100 80GB GPUs, 8 CPUs, 24 GiB requested | Running; startup verification |
| 2142046 | Combined report after both shards succeed | deep-chungus-6 | 2 CPUs, 8 GiB requested, no GPU | Pending on both evaluation jobs |

All three replacement jobs use account `grad-students`, partition `low-priority` and QoS `normal`. The exact commands, resource checks, cancellation and release times are in [the normal-priority submission receipt](math-sweep-15b-normal-submission-20260919.json). At 15:17:32 UTC both GPU jobs were running with no logged errors; GPU smoke and benchmark completion were not yet established.

These first replacement GPU jobs failed before generation. Their report dependency was cancelled; follow the version 2 jobs below. Do not resubmit the superseded 2142040 matrix. If either shard is preempted or fails, inspect its receipt and partial records before resuming the same shard. The combined report is dependent on both exact submitted job IDs; any replacement of those jobs also requires updating that dependency.

## Version 2 launch before occupancy recovery

Both first replacement jobs stopped before deriving or generating benchmark cells. Job 2142044 found another user's process using approximately 38 GiB on its Slurm-assigned `/dev/nvidia7`. A diagnostic allocation (2142047) confirmed that Slurm GPU IDs 6 and 7 map to device minors 6 and 7, while NVIDIA display indices differ. The other physically free GPU was `/dev/nvidia4`, assigned to the other Slurm job; it was not used. Only `/dev/nvidia6` was both allocated and physically free. Job 2142045 exposed an NFS shared-read-lock requirement: opening its descriptor write-only caused `EBADF`. Version 2 opens the shared lock in append/read mode.

The v2 launch uses one GPU on chungus-1 and two on chungus-11. It balances complete comparison groups by GPU count, with 60 cells / 15,175 responses on node 1 and 80 cells / 33,445 responses on node 11. The total remains 140 cells / 48,620 responses. Models, prompts, token limits, seeds, engine, and grading remain frozen. Twenty-five package tests passed, including actual GPU mapping, insufficient-free-memory rejection, one-GPU smoke sequencing, complete partitions, and the NFS-readable lock. This version did not reject an existing compute process when enough VRAM remained; the later occupancy recovery below closes that gap.

| Original v2 job | Role | Node | GPUs | Priority |
| --- | --- | --- | ---: | --- |
| 2142048 | Evaluation shard 0 | deep-chungus-1 | 1 | Normal |
| 2142049 | Evaluation shard 1 | deep-chungus-11 | 2 | Normal |
| 2142050 | Combined report after both succeed | deep-chungus-6 | 0 | Normal |

At 15:24:24 UTC both evaluation jobs were running in startup verification with no logged errors. This is not a claim of completed smoke tests or production generation. Failed jobs 2142044 and 2142045 are historical evidence only; their blocked report job 2142046 was cancelled.

- Frozen v2 package: [15b-math-normal-v2-20260919](../evaluation/launches/15b-math-normal-v2-20260919/README.md)
- Original v2 submission: [version 2 receipt](math-sweep-15b-normal-v2-submission-20260919.json)
- Source: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/evaluation-runs/15b-math-normal-v2-20260919/source`
- Output: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/outputs/math-sweep-15b-normal-v2-20260919`
- Launcher SHA256: `244f3e25f6dcd6108e93cb502251caf87e3cd2183e1f5088c7089362783b8f61`
- Archive SHA256: `a6a7e62baed46396662d63a48837cfd534621231bbe90aac4858151615f92bba`

The original v2 IDs were 2142048, 2142049 and 2142050; the current replacement IDs are listed below. A failed or preempted shard must resume its existing partition and saved responses; replacement job IDs require updating the report dependency. All earlier versions and original CPU preparation remain immutable for provenance. No training job was cancelled or changed in this evaluation replacement.

At 15:27:25 UTC, both v2 jobs passed the exact allocated-device and free-memory gates, completed all 140 derived preparations, and created smoke subprocesses. Node 1 uses `GPU-63dd215c-1936-6226-bfbf-a0f05a20d129`; node 11 uses `GPU-713f3893-df91-42ab-0810-d8b881f5bbed` and `GPU-8ceb7e35-48da-9616-c892-129f3109918c`. Both dispatch receipts record phase `smoke`, with zero completed smoke arms and zero production cells at that observation.


## Current normal-priority workflow: occupancy recovery

At 18:07 UTC, a read-only check inside node11 job **2142049** found `frankzydou` processes already using both assigned UUIDs: PID **2371117** on `GPU-713f3893-df91-42ab-0810-d8b881f5bbed` and PID **2371116** on `GPU-8ceb7e35-48da-9616-c892-129f3109918c`, 9,314 MiB each. The earlier VRAM threshold passed despite this overlap. The process cgroups identified only `slurmd.service`, so they did not establish which of the two foreign node11 jobs owned these PIDs. Only our conflicted evaluation **2142049** and its pending report **2142050** were cancelled. Node1 **2142048** continues, and every existing source, preparation, smoke output, partial production result and receipt is preserved.

The new [occupancy recovery package](../evaluation/launches/15b-math-normal-v2-occupancy-20260919/README.md) verifies the immutable v2 dispatcher and adds a startup gate rejecting every existing compute process on either assigned GPU, including stale processes owned by our user. It preserves the node11 A10080GB cohort, all 80 shard1 cells, complete comparison groups, prompt bytes, model revisions, decoding, seeds, grader, comparison identities and result paths. The guard identity and allocation evidence are recorded separately in dispatch provenance. Nine CPU tests passed; actual GPU and five-model smoke validation will run only after the replacement starts.

| Current job | Role | Node | GPUs | State at 18:14 UTC | Dependency |
| --- | --- | --- | ---: | --- | --- |
| 2142048 | Existing evaluation shard 0 continues | deep-chungus-1 | 1 | Running | None |
| 2142053 | Occupancy-guarded evaluation shard 1 | deep-chungus-11 | 2 | Pending | `afterany:2142020:2142041` |
| 2142054 | Combined report | deep-chungus-6 | 0 | Pending | `afterok:2142048:2142053` |

All three jobs use account `grad-students`, partition `low-priority`, QoS `normal`. The replacement GPU job waits conservatively for both current foreign node11 jobs to finish, then still requires an empty assigned-GPU process inventory before generation. Any occupancy failure stops without an automatic resubmission loop. The CPU report uses the unchanged v2 report implementation and requires all 140 cells to be complete and validated. No benchmark accuracy or base-versus-trained conclusion is reported at this stage.

- Reviewed commands, resource checks, job IDs and release verification: [occupancy recovery submission receipt](math-sweep-15b-occupancy-submission-20260919.json).
- New guard source: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/evaluation-runs/15b-math-normal-v2-occupancy-20260919/source`.
- Guard manifest SHA256: `bbf84a744d6de41b1cb36883d606a0ffef8eb584fc7e7b5c4a089144cd6062c2`.
- Guard archive SHA256: `fc969c85870f82f28e1c326d487f0af256668f9c853fe9261f9ecc5c2b474033`.
- Existing output remains `/mnt/nfs/home/mohamadzbib/projects/rl-infra/outputs/math-sweep-15b-normal-v2-20260919`.
- New allocation evidence: `results/.occupancy-recovery/JOB-RESTART/allocation.json`; per-cell and dispatch evidence remains in the existing v2 paths.

Monitor **2142048, 2142053 and 2142054** and the existing result receipts. Do not restart either shard or submit another report without inspecting these jobs and the recorded completed cells first.
