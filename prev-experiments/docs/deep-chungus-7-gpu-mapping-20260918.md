# deep-chungus-7 GPU mapping diagnosis — 2026-09-18

Job 2141895 runs Qwen2.5-Math-7B GRPO with staleness cap 4, normal priority, five A100 80GB GPUs, four trainer workers and one inference worker. The five-device BF16 backward and NCCL collective probe passed with sum 15. Model/data/tokenizer/custom-loss validation and the framework dry run passed. Production remains gated on a separate 25-update smoke.

## Confirmed cause of the misleading preflight warning

The launcher prints telemetry with `nvidia-smi -i "$CUDA_VISIBLE_DEVICES"`. On this node those integers are not interchangeable. Slurm is configured in `/usr/local/etc/gres.conf` with `AutoDetect=off ... File=/dev/nvidia[0-7]`. The job has Slurm device IDs 0–3 and 5. Driver `/proc/driver/nvidia/gpus/*/information` identifies the physical device minors separately from the NVIDIA monitoring indices.

| Slurm device file | NVIDIA monitoring index | PCI address | Assigned role |
|---|---:|---|---|
| /dev/nvidia3 | 0 | 0000:01:00.0 | Trainer |
| /dev/nvidia2 | 1 | 0000:24:00.0 | Trainer |
| /dev/nvidia1 | 2 | 0000:41:00.0 | Trainer |
| /dev/nvidia0 | 3 | 0000:61:00.0 | Trainer |
| /dev/nvidia5 | 6 | 0000:c1:00.0 | Inference |
| /dev/nvidia6 | 5 | 0000:a1:00.0 | Outside this job allocation |

The warning was from NVIDIA monitoring index 5, UUID GPU-6773861d-51c4-ced0-1784-de4d08064ea9. It returned Unknown Error for temperature and power. A query across all GPUs timed out after 15 seconds. The targeted query for the actual allocated devices, NVIDIA indices 0,1,2,3,6, returned ordinary temperature, power and memory readings. The inference device is UUID GPU-99e6d841-8f03-96c4-7573-c45c3e4026cc, not the device displaying the warning.

The job launcher reports inference on CUDA device 5 and trainers on 0,1,2,3. NCCL identifies the fifth probe rank at PCI c1:00.0 / NVIDIA index 6, confirming the actual compute mapping. The allocated physical set matches Slurm's requested device files; no missing or extra GPU was observed in the probe.

## Remaining uncertainty

The low-level cause of the unrelated device's telemetry error is not established. Kernel logs could identify an NVIDIA Xid or PCIe fault, but `dmesg` returned `Operation not permitted`. Do not infer a hardware failure or reset a GPU from these readings alone. No resets, driver changes, or changes to other jobs were performed.

Future telemetry should map Slurm device files to driver UUIDs and query `nvidia-smi` by UUID. Preserve the frozen running experiment and its manifests; the current misleading display does not change its verified compute allocation.

Evidence: `tmp/launch-7b-stale4-20260918/device-diagnosis.json`, `mapping3.raw.txt`, `status.raw.txt`, and `submission.json` in the parent workspace.
