# 7B staleness ≤6: five-GPU continuation on deep-chungus-9

## Superseded: launcher mapping error corrected

Native-device probe2142074 established that Slurm's original five visible GPUs were all idle. This package's extra translation from Slurm selectors to Linux device minors selected the occupied device incorrectly. The occupancy error was real for that wrongly selected device, but the earlier conclusion that the original five-GPU allocation could not run was incorrect. Job2142073 and its unnecessary dependency on2142035 were replaced by job**2142075**, normal priority on deep-chungus-9 with five GPUs and no dependency. See the [corrected native-CUDA launcher and evidence](../7b-stale6-native-cvd-20260920/README.md). The sections below preserve the prior attempt's history.

The user explicitly requested the five GPUs available on deep-chungus-9 at normal priority. This package replaces the six-GPU request with four trainer ranks and one inference replica, preserving the original experiment, checkpoint format, source/data manifests, training hyperparameters, evaluation schedule, run directories, and Comet project.

## Current queued continuation

At **01:38 UTC**, replacement job **2142073** was submitted and released with the same five-GPU normal-priority request on deep-chungus-9, resuming checkpoint125. Its dependency is `afterany:2142035`. It will become eligible after that conflicting job terminates, subject to resources and queue priority; the unchanged GPU occupancy and health gates still apply.

Read-only CPU diagnostic **2142072** confirmed the blocking process's parent chain ends in `slurmstepd: [2142035.batch]`. That job belongs to frankzydou. The earlier diagnostic2142071 expired waiting for a CPU allocation. No other user's workload was modified.

Slurm rejected `scontrol requeuehold 2142069` with `Invalid job id specified`, so one new replacement job was submitted. Job2142069 remains a failed startup attempt. The replacement is queued and has not restored the checkpoint or begun training.

See the [replacement receipt, exact command, process attribution and verified dependency](conflict-resubmission.json).

## Submission and startup

Job **2142069** requested five A100 GPUs, 48 CPUs and 384 GiB memory on **deep-chungus-9**, with account `grad-students`, partition `low-priority`, and QoS `normal`. The superseded six-GPU job **2142067** was cancelled after the replacement was submitted and verified.

On September 20 at **01:31:31 UTC**, the first attempt failed before model staging or checkpoint restoration. The prelaunch GPU gate found another process on an allocated physical device:

| Field | Observed value |
| --- | --- |
| GPU UUID | GPU-012e7790-9bf9-122d-b030-f88e489a373e |
| Device file | /dev/nvidia4 |
| PCI address | 0000:e1:00.0 |
| Process | 3248566 |
| Process owner | frankzydou, UID 28831 |
| GPU memory in use | 10,132 MiB |

A read-only CPU diagnostic, job **2142070**, verified process ownership and its parent chain to `slurmstepd`. No process was signaled. Slurm reported the other user's running job2142035 allocated GPU IDs5–7, while our job's five-device gate detected its process on physical minor4. The scheduler's allocation count alone therefore did not establish that five usable GPUs were available.

## Validation

Seven mapper tests, Python compilation and shell syntax checks passed locally. All 15 deployed package files matched their recorded hashes.

Inside job2142069, resolved-config and checkpoint validation passed. Trainer checkpoint125 and orchestrator next-step126 were confirmed, with model, optimizer and scheduler metadata present. Only allocation/inference topology fields changed from the original eight-GPU setup: total GPUs8→5, inference replicas/API workers4→1 and corresponding broadcast counts. Trainer world size remains four. The GPU health collective did not run because the occupancy check stopped startup first.

The configuration and checkpoint evidence support this continuation, but no training step has been completed by this five-GPU attempt.

## Evidence

- [Submission and effective Slurm settings](training-submission.json)
- [Config and checkpoint validation](validation.json)
- [Failed startup snapshot and exact error](startup-2142069.json)
- [Read-only process attribution](occupancy-diagnostic.json)
- [Frozen package manifest](package-manifest.json)
- [Derivation and unchanged inputs](provenance.json)

Package-manifest SHA256: `fad6d69cd79c91436476bcda15bc0a3683eb9ed77d6a191881574571359ea695`.

Remote package: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/recoveries/7b-stale6-five-gpu-20260920`.
