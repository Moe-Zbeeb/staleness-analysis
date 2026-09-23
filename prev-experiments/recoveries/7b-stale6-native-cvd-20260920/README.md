# 7B staleness ≤6 on five GPUs: preserve Slurm's CUDA assignment

Job **2142075** was submitted on September 20 at **01:53 UTC**, with account `grad-students`, partition `low-priority`, QoS `normal`, five A100 GPUs, 48 CPUs and 384 GiB memory on **deep-chungus-9**. It resumes checkpoint 125 with four trainer ranks and one inference replica. There is no dependency on another user's job. Superseded pending job 2142073 was cancelled after the replacement was submitted and verified.

## Startup state

At **01:57:18 UTC**, job **2142075** was RUNNING on deep-chungus-9 with all five GPUs, normal QoS and no dependency. Configuration, checkpoint metadata, source/data, tokenizer, GPU occupancy and the five-rank UUID/BF16/NCCL health checks all passed. The collective sum was 15 and all participant UUIDs matched the unchanged Slurm assignment. Model staging was underway; full checkpoint restoration and a new optimizer step were not yet verified. See [startup evidence](startup-2142075.json) and [GPU health results](gpu-health-2142075.json).

## Corrected diagnosis

The previous launcher incorrectly interpreted Slurm's CUDA-visible selectors as Linux device-file minor numbers and replaced the original visible devices. That remapping selected physical minor 4, occupied by frankzydou's job 2142035. This was our launcher error: the claim that Slurm's five visible GPUs were occupied was incorrect, and waiting for job 2142035 to end was unnecessary.

Diagnostic allocation **2142074 completed 0:0**. With Slurm's original `CUDA_VISIBLE_DEVICES=0,1,2,3,4` preserved and `CUDA_DEVICE_ORDER=PCI_BUS_ID`, all five selected GPUs were idle, each with 81,153 MiB free. Their physical device minors were 3,2,1,0,7; occupied minor 4 was outside that visible set. Slurm uses `task/affinity` and `proctrack/linuxproc` on this installation.

Slurm documents `CUDA_VISIBLE_DEVICES` as the job's GPU assignment and distinguishes CUDA/NVML PCI numbering from Linux device-file minor numbers. See the [official GPU management documentation](https://slurm.schedmd.com/gres.html).

## Launcher correction and validation

The launcher captures Slurm's initial CUDA-visible list before activating the runtime, preserves it unchanged, and verifies that the visible GPU UUIDs match their physical PCI identities. It never clears the CUDA-visible filter or substitutes selectors based on device-file minor numbers. PRIME's existing reordering assigns one of those five GPUs to inference and the other four to training without changing the device set.

The occupancy check still rejects an existing process on a selected GPU and requires at least 85% free memory. The five-rank health probe checks each actual UUID, BF16 backward computation and an NCCL all-reduce sum of 15. The original source, data manifests, four-shard checkpoint, optimizer/scheduler state, hyperparameters, evaluation schedule and Comet project remain unchanged. The four-trainer/one-inference overlay was already validated against checkpoint 125 and is revalidated at startup.

Nine local regression tests passed, including the observed node9 numbering mismatch, unchanged and noncontiguous selectors, duplicate UUIDs, PCI mismatch, modified CUDA visibility, existing processes and insufficient memory. Python compilation, shell syntax and all 17 remote package hashes passed.

## Evidence

- [Native-assignment probe](native-cvd-probe-2142074.json)
- [Submission, exact command and effective Slurm settings](training-submission.json)
- [Package manifest](package-manifest.json)
- [Configuration and derivation](provenance.json)

Package-manifest SHA256: `59afd9fc8bc01fceb8788f50f99a8411f3502dea0c1e5e34f1c06299693b9038`.

Remote package: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/recoveries/7b-stale6-native-cvd-20260920`.
