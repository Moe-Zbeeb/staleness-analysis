# deep-h-1 and deep-h-3: GPU failures and recovery drains

Checked September 20, 2026, at 15:38–15:40 UTC. All incident times below are UTC. Original Slurm output and some kernel captures use EDT (UTC−4).

## Finding

Both nodes experienced NVIDIA **Xid 79, “GPU has fallen off the bus,”** during previously healthy training runs. All eight GPUs on each node then received **Xid 154 with recovery action OS Reboot**. Training failed before the recovery drains were applied.

**Codex manually drained the nodes using sudo during the user's authorized recovery attempts**, then requested an OS reboot. The recorded drain reasons are those recovery actions. The captures do not show an automatic Slurm drain causing the training failures.

Neither node has been observed to re-register with Slurm after its final reboot request. Both currently report `DOWN+DRAIN+NOT_RESPONDING`, with zero allocated resources and no jobs on either node. A fresh check from `mslurm` found no ping reply and a TCP port 22 timeout for both nodes.

The confirmed failure is GPU loss from the driver's PCIe view. **The underlying hardware, power, PCIe, firmware, or driver trigger is not established. Neither is the reason the nodes did not return after the final reboot requests.** Those questions require console/BMC and additional administrator logs. NVIDIA's [Xid catalog](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html) describes Xid 79 as loss of GPU accessibility over PCI Express; the error alone does not identify which component caused it.

## Timeline

| Node | Time (UTC) | Verified event |
| --- | --- | --- |
| deep-h-3 | Sep 18 15:40:00 | Job 2141906 began. Eight-GPU BF16/NCCL health checks and a 25-update smoke passed; production completed five updates. |
| deep-h-3 | Sep 18 16:53:49 | Xid 79 at PCI `0000:c1:00.0`; all eight GPUs reported OS Reboot recovery. Inference reported CUDA unknown error at the same time. |
| deep-h-3 | Sep 18 16:56:52 | Job 2141906 ended `FAILED`, exit `1:0`. A later fresh diagnostic reproduced GPU C1 device-handle and CUDA initialization failures. |
| deep-h-3 | Sep 18 17:20:16 | Maintenance job 2141917 manually drained the node and requested reboot. |
| deep-h-3 | Sep 18 17:24–17:36 | Node returned, all eight GPU UUIDs were healthy with recovery action None, and the drain was removed. Replacement job 2141918 started at 17:27:03 and passed its eight-GPU BF16/NCCL probe at 17:36:59. |
| deep-h-1 | Sep 18 16:27:12 | Job 2141910 began. Eight-GPU health checks and smoke passed; production subsequently reached trainer step 106, with paired checkpoint 100. |
| deep-h-1 | Sep 18 21:11:07 | Xid 79 at PCI `0000:e1:00.0`; all eight GPUs reported OS Reboot recovery. |
| deep-h-1 | Sep 18 21:14:20 | Job 2141910 ended `FAILED`, exit `1:0`, after inference CUDA failure. |
| deep-h-1 | Sep 18 21:55:59–21:56:30 | Replacement job 2141963 failed its GPU health gate before training: seven devices requested Reboot and the remaining device did not return in the bounded inventory query. |
| deep-h-1 | Sep 18 21:57:06 | Maintenance job 2141964 manually drained the node and requested reboot. No successful return has been observed. |
| deep-h-3 | Sep 18 22:53:01 | A second Xid 79, now at PCI `0000:e1:00.0`; all eight GPUs again reported OS Reboot recovery. Job 2141918 had reached trainer step 145, with paired checkpoint 125. |
| deep-h-3 | Sep 18 22:56:13 | Job 2141918 ended `FAILED`, exit `1:0`. |
| deep-h-3 | Sep 19 01:31:39 | Maintenance job 2141969 manually drained the node and requested reboot. No successful return has been observed. |

## Exact final drain actions

The recovery scripts first verified that only their own maintenance allocation occupied the node, then ran:

```sh
sudo -n scontrol update NodeName=deep-h-1 State=DRAIN Reason=UserRequestedGPURecovery-20260919
sudo -n systemctl reboot --no-block
```

```sh
sudo -n scontrol update NodeName=deep-h-3 State=DRAIN Reason=UserRequestedGPURecovery-H3-20260919
sudo -n systemctl reboot --no-block
```

Both command captures printed `NODE_REBOOT_REQUESTED`. This confirms submission of the reboot request, not completion of boot.

The current Slurm reasons retain `root@2026-09-18T17:57:06` for h1 and `root@2026-09-18T21:31:39` for h3, in EDT. Slurm still retains their earlier boot times: September 18 15:20:11 UTC for h1 and 17:23:43 UTC for h3. The open accounting `DRAIN*` events begin later than the drain commands, at September 18 22:04:27 UTC and September 19 01:36:07 UTC respectively; these are separate recorded state transitions.

## Additional diagnostic clues and limits

- Both final failures were followed by Xid 74 NVLink fatal errors on PCI `c1:00` and Xid 119 GPU System Processor RPC timeouts. They are subsequent recorded errors, not proof of the initiating component failure.
- H3's second captured kernel excerpt also contains NVIDIA libos/RISC-V crash reports at 22:51–22:52 UTC, before its 22:53:01 Xid 79. An administrator should examine these with the driver/firmware versions; this excerpt does not establish a particular firmware bug.
- H1's capture contains an earlier PCIe AER Data Link Layer event on PCI `01:00.0` at 16:44:36 UTC. It is a different device from the later E1 loss; a causal connection is unproven.
- H3's first successful reboot and resumed training demonstrate temporary recovery, followed by recurrence. They do not establish permanent hardware damage or a software-only fault.
- Historical job accounting records application failures, not an OOM terminal state. Workload timing alone does not establish that the training configuration caused the underlying GPU loss.
- The controller log `/var/log/slurmctld.log` exists but the current read was denied. Current power, boot-screen and node-local post-reboot logs are unavailable through the unresponsive network path.
- This investigation made no cluster state changes and submitted no jobs.

## Message to the cluster administrator

> Could you check deep-h-1 and deep-h-3 through the management console/BMC? Both are currently DOWN+DRAIN+NOT_RESPONDING and do not respond to ping or TCP port 22 from mslurm. They had passed all-eight-H100 CUDA/NCCL checks and were training before NVIDIA GPU failures. H1 logged Xid 79, “GPU has fallen off the bus,” at PCI E1 on September 18 at 21:11:07 UTC. H3 first logged Xid 79 at PCI C1 at 16:53:49 UTC, recovered after a reboot and resumed training, then logged another Xid 79 at PCI E1 at 22:53:01 UTC. Each failure marked all eight GPUs as requiring an OS reboot, with NVLink fatal errors and GSP RPC timeouts also recorded. During our recovery attempts, we manually drained and requested reboot of h1 at September 18 21:57:06 UTC and h3 at September 19 01:31:39 UTC. Neither returned afterward. We can establish the GPU-loss failures and our drain/reboot actions, but not the underlying trigger or why boot/network access did not return. Please inspect power/boot state, BMC event logs, PCIe and GPU errors, and driver/GSP firmware evidence, then verify all eight GPUs before clearing the drains. Full incident logs are available.

## Evidence

Paths below are relative to the workspace root `/Users/mohammadzbeeb/Documents/ChatGPT/RL infra`.

- Current Slurm state, accounting and timezone: `tmp/h1-h3-drain-audit-20260920/current.json`.
- Current reachability and controller-log access: `tmp/h1-h3-drain-audit-20260920/reachability.json`.
- H1 kernel faults, exact drain reason and reboot acknowledgement: `tmp/recover-7b-stale8-h1-20260919/reboot.raw.txt`; executed command: `reboot.cmd` in the same directory.
- H3 first kernel faults: `tmp/sudo-recover-h3-20260918/reset-v2.raw.txt`; first drain/reboot: `reboot.raw.txt`; healthy return and production startup check: `recovery-result.json` in the same directory.
- H3 second fault, including preceding libos reports: `tmp/recover-7b-stale8-h1-20260919/h3-kernel.raw.txt`; the accompanying `h3-kernel.cmd` confirms this journal capture used `--utc`.
- H3 final drain and reboot acknowledgement: `tmp/recover-7b-stale6-h3-20260919/reboot.raw.txt`; executed command: `reboot.cmd` in the same directory.
- Training and recovery narratives: `staleness-analysis/docs/qwen7b-staleness6-h100-20260918.md` and `staleness-analysis/docs/qwen7b-staleness8-h100-20260918.md`.
