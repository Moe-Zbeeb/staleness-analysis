# 7B staleness ≤8 on deep-h-1

The user requested the next 7B staleness run on deep-h-1, continuing the immediately preceding normal-priority full-node H100 setup. Preflight found all eight H100 GPUs idle following the September 18 reboot.

Job **2141910** requests the full node exclusively: account `grad-students`, partition `h100`, QoS `normal`, eight H100 GPUs, 96 CPUs per task and 640G memory. Slurm verified the requested settings before the job was released. At 16:27:22 UTC, the job was RUNNING on deep-h-1 with all eight GPUs allocated; startup validation was still in progress.

Topology is four trainer GPUs and four inference replicas. The run starts from pinned Qwen2.5-Math-7B, retaining the baseline dataset, tokenizer, reward, optimizer, batch size 64, group size 8, 4,096-token context, 3,072-token completion budget and seed 42. Staleness cap is eight. Hardware and topology differences from the original cap-2 baseline are declared in the comparison manifest. The separate cap-6 H100 experiment uses the same topology.

Local static checks and cluster-resolved configuration checks passed. A bounded NVIDIA inventory and eight-rank BF16/NCCL test precede preparation. A separate 25-update smoke must pass before fresh 1,000-update production. The production Comet project is `qwen25-math7b-grpo-stale8-8h100`.

Local receipts and state: `tmp/launch-7b-stale8-h1-20260918`. Frozen implementation: `staleness-analysis/experiments/dapo-qwen25-math7b-grpo-stale8-8h100`. Remote experiment: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/experiments/dapo-qwen25-math7b-grpo-stale8-8h100`.

Source-manifest SHA-256: `c2473d921af7653116a4dae9e749651334c6a4f6b4ec7f11f0f76f4914a5d682`.

At 2026-09-18T16:30:18.648431+00:00, all eight allocated H100 GPUs passed UUID matching, BF16 backward and NCCL all-reduce (sum 36, world size eight). The job remained RUNNING and advanced to preparation for its 25-update smoke. Production updates and Comet ingestion are not yet verified.

## Failure identified September 19 (Beirut)

At 21:52 UTC September 18, deep-h-1 was IDLE and job 2141910 had ended FAILED with exit 1:0. Accounting records end time 17:14:20 in cluster local time, equivalent to 21:14:20 UTC September 18 or 00:14:20 Beirut September 19.

Production completed trainer step 106. The latest paired checkpoint is step 100, with trainer metadata, four nonempty shards, and orchestrator progress present. The step-100 trainer save completed at 20:59:54 UTC. This is structural checkpoint evidence; recovery loading has not yet been tested.

The inference log contains `torch.AcceleratorError: CUDA error: unknown error` in worker 180510, followed by API server failures. The trainer then waited for its broadcast receiver, and the launcher terminated the run after inference exited with code 1. Earlier router connection failures also appear around 20:59:50 UTC, near checkpoint completion; their causal relationship to the later CUDA failure is not established. Do not infer an out-of-memory condition or repeat of deep-h-3's Xid79 without GPU/kernel diagnostics.

No restart, GPU reset, or reboot was performed during this status check. Evidence: `tmp/cluster-health-20260918/h1_idle.raw.json` and `h1_first_error.raw.json`.

## Authorized recovery attempt

The user requested recovery after the failure report. Replacement job 2141963 retained normal QoS, exclusive eight-H100 allocation on deep-h-1, the frozen experiment, and checkpoint 100. Before entering the existing training wrapper, a bounded NVIDIA inventory and eight-rank health gate were added in a separate recovery wrapper. NVIDIA reported OS reboot recovery action for the seven responding GPUs; the inventory then timed out. The training wrapper did not run and existing checkpoint state was preserved.

Maintenance job 2141964 acquired all eight GPUs exclusively. Sudo verified that this was the only RUNNING/COMPLETING job on deep-h-1. The kernel recorded Xid 79 at PCI e1:00 at 21:11:07 UTC: GPU fallen off bus. All eight devices received Xid 154 recovery action OS Reboot; PCI c1:00 subsequently recorded NVLink Xid 74 and GSP RPC timeout Xid 119. These logs establish the immediate device/PCIe failure, not its underlying physical or driver cause.

At 21:57:06 UTC the node was drained with reason `UserRequestedGPURecovery-20260919`, then rebooted through sudo. The maintenance job completed. Re-registration, post-boot GPU health, and training resumption remain to be verified. Other nodes and their jobs were not changed. Audit files: `tmp/recover-7b-stale8-h1-20260919/`.

At 22:05:45 UTC (01:05:45 Beirut September 19), roughly nine minutes after the reboot request, deep-h-1 still did not respond to ping and Slurm reported DOWN+DRAIN+NOT_RESPONDING. No post-boot training job has been submitted or released. Recovery is blocked on node availability; the next step is a management-console or administrator check of its boot state. The prepared post-reboot restart remains unexecuted. Final local receipt: `tmp/recover-7b-stale8-h1-20260919/recovery-result.json`.

At 22:13:02 UTC September 18 (01:13:02 Beirut September 19), the node remained DOWN+DRAIN+NOT_RESPONDING approximately 16 minutes after the reboot request. The user requested further sudo recovery. Sudo on mslurm explicitly denied this account; no additional privileged attempts were made there. The lab administration guide and visible machine-failure records were read, and accessible management configuration was inspected. No BMC endpoint or remote-power configuration was found in those sources. The remaining recovery requires the node management console or physical access. Checkpoint 100 and other running jobs remain unchanged. Evidence: management-sudo.raw.txt, management_inventory.raw.json, and boot_status.raw.txt in the recovery audit directory.
