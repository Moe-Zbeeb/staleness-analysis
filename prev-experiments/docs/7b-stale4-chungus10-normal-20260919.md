# 7B staleness ≤4 continuation on chungus-10

The user requested that Qwen2.5-Math-7B staleness ≤4 use the five remaining GPUs on `deep-chungus-10` at normal priority, alongside the existing three-GPU high-priority 3B continuation.

## Submission

- Training job: **2142029**.
- Account / partition / QoS: `grad-students / low-priority / normal`.
- Placement: `deep-chungus-10`, five A100 GPUs, 48 CPUs, 384 GiB host memory, no exclusive-node request.
- Immediately before submission, the node had three GPUs and 32 CPUs allocated to 3B job **2142027**, leaving five GPUs and sufficient CPU and host memory for this continuation.
- Resume checkpoint: **225**, with four complete trainer shards and orchestrator next-step 226.
- Previous job **2141895** had been preempted and then failed during its automatic restart. This submission does not duplicate an active or queued 7B cap-4 job.
- Incomplete checkpoint 250 is preserved in the existing recovery directory before selecting the paired checkpoint 225.

## Preserved experiment

The original five-GPU topology remains four trainer GPUs plus one inference GPU, TP1/DP1, one API process. There is no configuration overlay. Frozen experiment files, data/model/tokenizer hashes, optimizer/scheduler, sampling, staleness bound, evaluation protocol, output paths and Comet identity remain unchanged.

The replacement launcher adds explicit Slurm-device-to-CUDA-UUID mapping, a three/five-job separation check through per-allocation UUID health verification, immutable resume capture, clear run-lock failure reporting and refusal to start from scratch. The final audit retains the original configuration checks and accepts pruned resume checkpoints only with verified captured provenance and successful resume/completion logs.

## Validation and provenance

CPU validation job **2142028** completed with exit `0:0`, verifying the original smoke gate, unchanged resolved GPU settings, all four checkpoint shards and model/optimizer/scheduler/progress state. Eight recovery guard tests and four device-mapping tests passed locally, including complementary 3+5 allocations under a permuted CUDA ordinal order.

Actual GPU startup and resumed training are verified separately after submission. The health probe checks every allocated UUID before five-rank BF16 backward and NCCL all-reduce; the expected sum is 15. Slurm GRES configuration is bound to SHA-256 `1a5c2624ff52be8400be632e1aed6f61b75399446b91d9eb249d19ac38307732`.

- Recovery package: `recoveries/7b-stale4-five-gpu-20260919/`.
- Package manifest SHA-256: `929e10f92f3746c5ce546da41371da181f8ab7042f4c79bbac69fc6a9ac5610c`.
- Original source manifest: `23bac478489812f972b419f6b01a5a375521b8088bffc9b6a722fb1a7054e7b8`.
- Original data manifest: `73afcc50c0b07eec405a88d576344f01b0060a2d32f486bf804c15bfd45dcc15`.
- Checkpoint trainer metadata: `b0eead4d167aad3bf95892074b7d24c1b4a35ca859fcc5f3875405ea2bcb4059`.
- Checkpoint orchestrator progress: `afb787af4fa5384e864e603511147de77393e91d561551abbbdc35268b61c560`.

Remote receipts, per-attempt GPU mappings and checkpoint captures are under `/mnt/nfs/home/mohamadzbib/projects/rl-infra/recoveries/7b-stale4-five-gpu-20260919/`. Local inspection/submission evidence is under `tmp/resume-7b4-five-gpu-20260919/`.

## GPU startup verified

Job **2142029** started on `deep-chungus-10`. The five-rank GPU health check passed with NCCL sum 15, finite BF16 backward, and exact allocated UUIDs. Canonical CUDA selection is `0,7,6,5,4`; Prime receives inference GPU `4` and trainer GPUs `0,7,6,5`. The checkpoint 225 capture exists.

The two allocations are disjoint and together cover all eight GPU UUIDs and CUDA ordinals. Startup took several minutes while processes waited on shared-storage reads; the health check passed within its existing bound, so no timeout or health assertion was changed. Full restoration and subsequent training are distinct from the completed GPU health check.

At 11:49:42 UTC, Slurm reported both jobs running. Both Prime launchers had completed with the expected disjoint GPU assignments, and both Comet bridges were connected. Saved-state initialization was still underway; a new completed training step had not yet been observed.
