# 7B staleness ≤6 on deep-h-3

The user explicitly requested normal priority and all GPUs of deep-h-3. Live inventory showed an idle node with eight H100 GPUs, 128 CPUs and 1,000,000 MB configured RAM following a reboot. The previous A100 job 2141902 had been preempted before creating smoke or production artifacts; it was held and then cancelled after the replacement submission was verified.

Job **2141906** requests the full node exclusively, `grad-students` account, `h100` partition, `normal` QoS, eight H100s, 96 CPUs per task and 640G memory. The live running allocation accounted for all 128 node CPUs and all eight GPUs.

Topology is four trainer GPUs and four inference replicas. Fresh training starts from pinned Qwen2.5-Math-7B, with the baseline dataset, tokenizer, reward, optimizer, batch size 64, group size 8, 4,096 context, 3,072 completion budget and seed 42. Staleness cap is six. Hardware and topology differences are declared in the frozen comparison manifest.

Static source/settings and cluster-resolved configuration checks passed. A bounded NVIDIA inventory and eight-rank BF16/NCCL test precede preparation. A separate 25-update smoke must pass before fresh 1,000-update production. The production Comet project is `qwen25-math7b-grpo-stale6-8h100`.

At 2026-09-18 15:40:39 UTC, Slurm reported RUNNING on deep-h-3 and both CUDA_VISIBLE_DEVICES and SLURM_JOB_GPUS exposed 0 through 7. The next monitoring connection failed at SSH multiplexed forwarding with Broken pipe, so automated reconnection attempts stopped. GPU computation, smoke completion and production training are not yet verified.

Local receipt and state: `tmp/launch-7b-stale6-h3-20260918`. Frozen implementation: `staleness-analysis/experiments/dapo-qwen25-math7b-grpo-stale6-8h100`. Remote experiment: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/experiments/dapo-qwen25-math7b-grpo-stale6-8h100`.

A later authorized connection succeeded. At 2026-09-18T16:28:45.256071+00:00, the job remained RUNNING. All eight H100 devices passed allocation UUID verification, BF16 backward and NCCL all-reduce (sum 36). The smoke trainer completed update 25 with finite logged metrics, and the four-question smoke evaluation completed. Final checkpoint saving and the smoke audit were still pending; production completion is not established.

## Production failure, September 18

At 16:53:49 UTC (19:53:49 Beirut), vLLM worker PID 111968, inference replica DP2, raised `torch.AcceleratorError: CUDA error: unknown error` while synchronizing `async_copy_ready_event` in `gpu_model_runner.py:313`. This is the first fatal error in the inference log, beginning at line 12570. The same stack explicitly warns that CUDA errors can be reported asynchronously, so it does not identify the originating kernel or prove a hardware versus software cause.

DP2 then failed; API servers reported EngineDeadError and HTTP 500 responses. The parent inference process exited with code 1, PRIME terminated the remaining processes, and Slurm recorded job 2141906 FAILED with exit 1:0 at 16:56:52 UTC (19:56:52 Beirut). The node returned to IDLE with zero allocated GPUs. The recorded terminal state is application failure; Slurm does not report preemption or an out-of-memory terminal state.

Both eight-GPU BF16/NCCL checks and the complete 25-update smoke audit had passed. Production completed five trainer updates with finite losses and gradients before this failure. Only an orchestrator progress file exists under production checkpoint step_6; there is no paired production trainer checkpoint to resume. The separate successful smoke checkpoint remains validation evidence and must not be counted as production progress.

Evidence: `tmp/diagnose-7b-stale6-h3-20260918/status.raw.json`, `errors.raw.json`, and `first-failure.txt`. Kernel/Xid evidence was not collected, so the underlying cause of cudaErrorUnknown remains unresolved. No automatic replacement job was submitted during this diagnostic check.

## Recovery check: persistent GPU/driver fault

After the user restored SSH, diagnostic job **2141911** ran exclusively on all eight H100 GPUs of deep-h-3 with normal QoS on the h100 partition. It failed in 32 seconds at 17:12:09 UTC, before any model loading or training.

NVIDIA reported: `Unable to determine the device handle for GPU6: 0000:C1:00.0: Unknown Error`. The earlier healthy inventory identifies that PCI device as UUID `GPU-fcb12563-5061-4da3-e780-d919cda2b8be`. In the failed training run, inference used devices 4,5,6,7, and the first failed worker was inference replica DP2, corresponding to this device under that observed mapping.

Fresh PyTorch processes in the full eight-device allocation then failed CUDA initialization, set available devices to zero, and raised `ProcessGroupNCCL is only supported with GPUs, no GPUs found!`. This establishes a persistent device/driver accessibility problem after the training process exited. It does not prove a permanently defective card, the originating kernel fault, or whether a reset/reboot will suffice. Both journalctl kernel access and dmesg were denied to this account.

The diagnostic ended and released its allocation. No speculative vLLM configuration change, GPU reset, driver change, or replacement training job was performed. The original failed run and successful smoke evidence remain preserved. Administrator investigation/recovery of the device is required to restore the requested eight-GPU run on this node.

Evidence is in `tmp/recover-7b-stale6-h3-20260918/state.json`, `check.json`, and `check-2141911.err`. Remote diagnostic directory: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/diagnostics/7b-stale6-h3-20260918`.

## Authorized sudo recovery

The user explicitly authorized sudo recovery. An authenticated sudo check in full-node allocation 2141914 confirmed `(ALL) ALL` on deep-h-3 and uid 0. The noninteractive probe 2141912 had only established that a password was required, not that the user lacked privileges. A helper misclassified the known `/run/user/29562` mkdir warning in 2141913; that allocation was confirmed cancelled before another attempt.

Privileged kernel inspection in 2141916 established **Xid 79: GPU PCI 0000:c1:00 fell off the bus** at 16:53:49 UTC. All eight devices then reported Xid 154 with recovery action **OS Reboot**. PCI e1:00 additionally reported Xid 74 NVLink errors and Xid 119 GSP RPC timeouts. The only remaining GPU client was root's NVIDIA persistence daemon. A targeted reset was therefore not executed; the driver's explicit reboot recovery action was followed.

In exclusive allocation 2141917, a fresh Slurm check confirmed that only the recovery job occupied deep-h-3. The node was drained with reason `UserRequestedGPURecovery-Xid79-20260918`, then `sudo systemctl reboot` was issued at approximately 17:20:16 UTC. Other nodes and jobs were untouched. Post-reboot verification and production restart are pending.

Evidence: `tmp/sudo-recover-h3-20260918/sudo-auth-v2.raw.txt`, `reset-v2.raw.txt`, and `reboot.raw.txt`. Root logs were copied from node-local storage to `/mnt/nfs/home/mohamadzbib/projects/rl-infra/diagnostics/sudo-recover-h3-20260918/h3-gpu-recovery-2141916` using the regular user account because the shared NFS filesystem rejects root writes.

## Recovery and relaunch verified

The reboot restored all eight H100 devices. Direct read-only maintenance inspection through the CSAIL jump host at about 17:25:33 UTC showed uptime around one minute, boot ID `a9dba3dd-0872-49aa-85d2-f36575df7e87`, active slurmd, and all eight known GPU UUIDs with `gpu_recovery_action=None`, including PCI C1:00.0. No Xid entries were returned from the new boot's kernel log. The RPC security service rpc-svcgssd was in failed state after boot; it was left unchanged. Shared-storage reads and job execution were working.

The recovery drain was removed after verifying its exact reason and the eight-device inventory. Slurm re-registered the node as IDLE with new boot time 17:23:42 UTC and slurmd start 17:24:36 UTC.

Replacement production job **2141918** was submitted with the original frozen experiment: account grad-students, h100 partition, normal QoS, exclusive eight H100s, four trainer plus four inference GPUs, 96 requested CPUs, 640G memory, and unlimited duration. The original submission receipt remains unchanged; the new receipt is `recovery-submission-20260918.json` in the experiment directory. No training hyperparameters or vLLM settings were changed. The wrapper revalidates the existing passed smoke, preserves failed production artifacts, and starts fresh production because no paired production checkpoint exists. All-eight-GPU CUDA/NCCL startup verification is pending.

This is recovery from the observed fault, not proof that the underlying PCIe/device problem cannot recur. Sources: local `tmp/sudo-recover-h3-20260918/node-check.raw.txt`, `resume-node.raw.txt`, `restart.json`, and `training-state.json`.

At **17:36:59 UTC**, the replacement job's built-in eight-rank health probe passed: all eight expected UUIDs matched the allocation, BF16 backward succeeded, and NCCL all-reduce returned 36 with world size eight. Device rotation exposed inference 4–7 and trainers 0–3. The production launcher entered fresh training initialization at 17:36:51 UTC and Comet logging started. A production optimizer update is not yet verified at this snapshot.

A supplementary diagnostic step in the same exclusive allocation stopped earlier on a missing `SLURM_JOB_GPUS` environment variable (srun step versus batch environment), after CUDA/NCCL initialization. That diagnostic is not counted as a hardware failure or a passing test. The untouched batch workflow subsequently supplied the correct Slurm variables and passed its complete health probe. Receipt: `tmp/sudo-recover-h3-20260918/recovery-result.json`.

## September 19 recurrent failure diagnosis

Checked at 01:26 UTC September 19. Job 2141918 failed at 22:56:13 UTC September 18, after trainer step 145. Paired checkpoint 125 contains trainer metadata, four nonempty distributed shards (about 91.4 GB combined), and orchestrator progress.pt; reload has not been tested. The inference worker reported CUDA unknown error at 22:53:01 UTC. Sudo kernel-journal inspection confirms Xid 79 at PCI e1:00 (GPU fallen off bus) at that same time, Xid 154 requiring OS Reboot on all eight devices, followed by NVLink Xid 74 and GSP timeout Xid 119 on c1:00. SSH and slurmd still work; Slurm incorrectly remains IDLE with respect to actual GPU health. A bounded nvidia-smi query returned seven devices requiring Reboot and did not return the eighth. This check performed no reboot or resubmission. Evidence: tmp/recover-7b-stale8-h1-20260919/h3_current.raw.json, h3_failure.raw.json, h3-node-check.raw.txt, h3-kernel.raw.txt.

The user authorized recovery and restart on h3. Maintenance job 2141969 acquired all eight H100 GPUs exclusively at normal priority, confirmed that no other running/completing job occupied the node, drained it with reason `UserRequestedGPURecovery-H3-20260919`, and issued `sudo systemctl reboot --no-block` at 01:31:39 UTC September 19. The command returned `NODE_REBOOT_REQUESTED`. Checkpoint125 is preserved; post-boot validation and restart are pending. A separate recovery wrapper will require the expected eight GPU UUIDs and recovery action None, followed by the unchanged production CUDA/NCCL health probe. Audit: `tmp/recover-7b-stale6-h3-20260919/`.

At 01:37:44 UTC September19, replacement job2141971 was submitted and released at normal QoS, account grad-students, partition h100, exclusive8H100 on deep-h-3 with96CPUs and640G. Source manifest SHA256 is326cb17a9225e3aa8c29b17343ca72f7f5a79172ffbc15306d3421b751fd98cf; data manifest SHA256 is15de7faf057da17928a625d05ddc7ea8f8c1485e7de20cdafd8a6030a5f6de68. Both remain identical to the saved run. The job is pending node recovery; no GPU probe, checkpoint load, or new training step has run. Remote receipt: experiments/dapo-qwen25-math7b-grpo-stale6-8h100/recovery-post-reboot-submission-20260919.json.

At 01:40:11 UTC September 19, approximately8.5minutes after the reboot request, h3 remained unreachable by ping and TCP22, and Slurm reported DOWN+DRAIN+NOT_RESPONDING with its previous boot timestamp. Job2141971 is released but PENDING because its requested node is down/drained. No post-boot health verification, checkpoint load, or training step has occurred. The documented fallback is the lab IPMI management hostname/login through holyoke-console.csail.mit.edu, per https://tig.csail.mit.edu/hardware/ipmi/; those separate credentials and h3 endpoint are unavailable in this task. Retain the queued job and do not duplicate it. Final local receipt: tmp/recover-7b-stale6-h3-20260919/recovery-result.json.
