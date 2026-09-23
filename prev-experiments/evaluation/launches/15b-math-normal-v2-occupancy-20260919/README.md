# Node11 occupancy recovery for the 1.5B math sweep

This new launcher resumes only the original node11 shard: 80 production cells, all five models, normal priority, two A10080GB GPUs on `deep-chungus-11`. It preserves the immutable v2 dispatcher, original preparation, derived preparation, result paths, comparison identities, prompts, generation/scoring code and configuration. Node1 job `2142048` continues independently with its existing 60 cells and completed responses.

The v2 guard required enough free VRAM but did not reject another process using the same physical GPU. At 18:07 UTC, the UUIDs allocated to job `2142049` still contained `frankzydou` PIDs `2371116` and `2371117`. The new guard rejects every pre-existing compute process on either assigned UUID, even when free VRAM exceeds the frozen 85% reservation. It records the allocation and outcome under `results/.occupancy-recovery/JOB-RESTART/allocation.json`. The new guard manifest hash and receipt hash are included in the dispatch allocation provenance. A blocked allocation exits; there is no automatic resubmission loop.

The wrapper imports and verifies the exact frozen v2 dispatcher (`244f3e25f6dcd6108e93cb502251caf87e3cd2183e1f5088c7089362783b8f61`) and wraps only its allocation gate in memory. It runs its unchanged partition, derivation, model smoke checks, duplicate locks and production workers with the original launcher identity. Thus existing preparations/results are reused, and the original combined report remains valid. The additional wrapper identity is recorded separately; it is not presented as an unchanged execution launcher.

## Submission

Deploy this directory as a new immutable package at:

`/mnt/nfs/home/mohamadzbib/projects/rl-infra/evaluation-runs/15b-math-normal-v2-occupancy-20260919/source`

The submit script requires the parent to verify the two foreign jobs before choosing the dependency. Their process cgroups identify only `slurmd.service`, not a specific job. The conservative reviewed choice is to wait for both node11 jobs `2142041` and `2142020` to end:

```bash
python3 /mnt/nfs/home/mohamadzbib/projects/rl-infra/evaluation-runs/15b-math-normal-v2-occupancy-20260919/source/submit.py \
  --expected-manifest-sha256 REVIEWED_PACKAGE_SHA256 \
  --wait-for-job 2142041 --wait-for-job 2142020
```

Effective GPU request: account `grad-students`, partition `low-priority`, QoS `normal`, node11, two A100 GPUs, eight CPUs, 24 GiB host memory, 24 hours, requeue, no exclusive allocation. Dependency: `afterany:2142041:2142020`. The user explicitly authorized normal priority and available partial-node GPUs for this evaluation. The script checks fresh Slurm resources and owners, verifies original jobs `2142049` and `2142050` have been cancelled, refuses duplicates, submits held jobs, records each ID and command immediately, verifies effective fields/dependencies, then releases them.

The replacement CPU report uses the unchanged v2 `report_job.sh` on `deep-chungus-6`, normal priority, zero GPUs, two CPUs, 8 GiB, one hour, with `afterok:2142048:NEW_SHARD_ID`. It still requires all 140 original cell identities exactly once and complete validated results before reporting. Both jobs and their held/released verification evidence are recorded in `evaluation-runs/15b-math-normal-v2-occupancy-20260919/submission.json`.

A pre-existing submission receipt prevents any repeated submission, including ambiguous failures. Inspect the receipt and scheduler before any manual recovery. No script cancels other jobs or touches another user's processes.

## Node1 fallback feasibility

All five model arms already passed native-context smoke checks on node1 A10040GB hardware. Its fixed single-GPU BF16 engine, native 4,096 context, fixed batch/concurrency and token budgets can therefore run shard1 without a model/configuration change. The frozen dispatcher intentionally does not support that reassignment: node11 shard1 currently binds A10080GB hardware and node11 into comparison and cell identities.

A future move requires a separate 80-cell derivation with A10040GB/node1 identities, copying prompt bytes exactly and retaining every full `(profile, benchmark)` group with all five model arms and budgets in the same new cohort. No node11 production samples should be mixed into those comparisons. A new combined report must validate the 60 unchanged v2 shard0 cells plus exactly those 80 newly derived shard1 cells, with no duplicate original IDs. Merely changing the shard's node or minimum-memory constants would recompute partitions/identities and invalidate the existing report. This package deliberately keeps node11 and does not implement migration.

## Local validation

```bash
local-envs/rlsc-cpu-tests/bin/python -m unittest discover \
  -s staleness-analysis/evaluation/launches/15b-math-normal-v2-occupancy-20260919 \
  -p 'test_*.py' -v
bash -n staleness-analysis/evaluation/launches/15b-math-normal-v2-occupancy-20260919/evaluate_job.sh
```

Tests reproduce the original gate accepting the observed 9.3 GiB competing process and verify the new guard rejects it, including same-user stale processes. They verify full UUID/cohort inventory, durable pass/failure provenance and the unchanged frozen source manifest. Actual CUDA/model smoke validation is performed within the future allocation; local tests do not prove live GPU availability.
