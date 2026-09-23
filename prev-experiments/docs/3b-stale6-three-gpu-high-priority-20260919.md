# 3B staleness ≤6 continuation on three high-priority GPUs

The user requested that the existing Qwen2.5-3B staleness ≤6 production run continue on three GPUs at high priority. Job **2142027** replaces the pending normal-priority job **2141859**. The high-priority account cap is 12 GPUs; the existing 14B job uses nine.

## Continuation and scheduling

- Resume checkpoint: **775**, with both trainer shards, trainer metadata, optimizer, scheduler and orchestrator state verified.
- The preempted attempt completed trainer step 799. Checkpoint 800 has no trainer metadata and is incomplete; the launcher preserves it in the existing recovery directory before selecting checkpoint 775.
- Account / partition / QoS: `grad-students / high-priority / high-priority`.
- One node, three A100 GPUs, 32 CPUs and 160 GiB host memory; no exclusive-node request.
- Eligible 80 GB A100 nodes: `deep-chungus-7`, `deep-chungus-9`, `deep-chungus-10`, `deep-chungus-11`.
- The replacement was submitted held, its effective scheduling fields verified, the old pending job cancelled, and the replacement released. The 14B allocation was preserved.

## Execution change

The two trainer ranks are unchanged. Inference uses one GPU with tensor parallelism 1 instead of two GPUs with tensor parallelism 2. Data parallelism and API-server count remain 1. The exact resolved configuration changes are:

| Field | Before | Continuation |
| --- | --- | --- |
| Total GPUs | 4 | 3 |
| Inference GPUs | 2 | 1 |
| Inference tensor parallelism | 2 | 1 |
| Trainer broadcast inference world size | 2 | 1 |
| Orchestrator broadcast inference world size | 2 | 1 |

The frozen experiment source, dataset hashes, model, tokenizer, training loss, optimizer, sampling settings, evaluation protocol, sequence length, staleness bound and run/Comet identity remain the same. This execution segment changes inference capacity and may change the observed asynchronous staleness distribution; record it when comparing runs.

## Device selection and validation

The launcher verifies the reviewed Slurm GRES configuration and resolves the allocated `/dev/nvidia` files to GPU UUIDs. It queries CUDA ordinals with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, then selects precisely those UUIDs. This avoids assuming that Slurm device minors and CUDA numeric ordinals are identical. The health probe checks every UUID before BF16 backward and an NCCL all-reduce across all three GPUs. Prime receives one inference GPU followed by the two trainer GPUs.

CPU validation job **2142026** completed with exit `0:0`. It verified the five exact resolved differences, the unchanged two-rank trainer, checkpoint step 775, orchestrator next-step 776, and retained model/optimizer/scheduler/progress state. Fifteen local recovery and device-mapping regression tests passed. GPU validation and resumed training are recorded separately after startup.

The recovery launcher captures immutable checkpoint metadata and prelaunch resolved configurations before Prime overwrites runtime configuration files. The final audit permits checkpoint retention only with matching capture, successful resume logs, and newer complete checkpoints. The launcher refuses a fresh restart when a valid paired checkpoint is missing.

## Provenance

Recovery package: `recoveries/3b-stale6-three-gpu-20260919/`.

- Package manifest SHA-256: `63df3a5bc2a0cdf9cddda6bdbe83406929c56c1c26fe6c7c8485d622fd6a56f1`.
- Original source manifest SHA-256: `cbd27323390ea2e6c9176cd6e0d7186ae54ffe126a1da18bc321d6d863742e92`.
- Original data manifest SHA-256: `d5e2b591d2153b44924eaf63788dbe1f18149c32df5cffdca384957f6116d169`.
- Checkpoint 775 trainer metadata SHA-256: `527ba554ecb816ca6feffb8abd4fae203c9f9a5a0ab92cebd13707a9ca008178`.
- Checkpoint 775 orchestrator progress SHA-256: `ff10ceaffab9b843d24e0d6bf3ade735a5aaffa7788b34b1adf7e4aee79b6d89`.

Remote receipts and mapping/capture records are under `/mnt/nfs/home/mohamadzbib/projects/rl-infra/recoveries/3b-stale6-three-gpu-20260919/`. Local inspection and submission records are under `tmp/resume-3b6-three-gpu-20260919/`. No source files in the frozen experiment were modified.

## GPU startup verified

Job **2142027** started on `deep-chungus-10`. The three-rank GPU health check passed with NCCL sum 6, finite BF16 backward, and exact allocated UUIDs. Canonical CUDA selection is `3,2,1`; Prime uses inference GPU `1` and trainer GPUs `3,2`. The checkpoint 775 capture exists, and inference broadcast initialization returned HTTP 200.

At 11:48:09–11:48:19 UTC, both trainer processes held checkpoint 775 shard files open and increased their read counters by 376,445,824 bytes each. This confirms checkpoint loading was progressing. No resumed training step had yet been observed at that inspection.

At 11:49:42 UTC, Slurm reported both jobs running. Both Prime launchers had completed with the expected disjoint GPU assignments, and both Comet bridges were connected. Saved-state initialization was still underway; a new completed training step had not yet been observed.
