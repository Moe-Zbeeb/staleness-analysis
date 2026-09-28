# DAPO 1.5B live timing investigation

At 2026-09-28 11:43:54 UTC, learner job **2145435** on deep-chungus-5 had no committed update. Its first rollout cohort was complete, and the learner was still processing it. Historical job **2145436** on deep-chungus-3 had already completed the separate policy-zero cohort needed at update 257. The leading bottleneck is communication within the four-GPU learner. This is a diagnosis from passive measurements, not an exclusive CUDA operator profile or a measured optimization speedup.

[Machine-readable evidence](../diagnostics/dapo-live-profile-20260928.json) records the timings, GPU samples, socket counters and archive packing results. The active frozen source remains `cb27c5b`. Training settings, running processes and upstream library files were not changed.

## Where the elapsed time went

| Stage | Observation | Interpretation |
| --- | --- | --- |
| Allocation to first generation | Approximately 10 min 31 s | One-time staging and startup, including inference readiness; not recurring update time |
| Initial policy export | 12.06 s after receiver acknowledgement | Included in startup above; excludes waiting for inference to become ready |
| Main on-policy cohort | 208.80 s, or 3 min 29 s | 512 responses, 2,883,220 generated tokens; includes collection, grading and archival work |
| Packing | 5.64 s in a separate CPU replay | Repacking the actual immutable archive, not a timer from the original live operation |
| After generation, before update completion | More than 27 min 36 s at the snapshot | This is an unfinished interval, not a completed learner duration; includes packing/delivery and learner work |
| Historical cohort on the other node | 187.78 s, or 3 min 8 s | Runs independently; do not add it to the main critical path |
| Token archive sealing, policy-one export, recovery checkpoint | Not reached at the snapshot | No measurement available yet; these cannot explain the current long microbatch phase |

The main cohort averages 5,631 output tokens per response. It has 370 truncated responses out of 512 (72.3%) and mean reward 0.2480. The smaller dataset reduces dataset size, but the configured update still processes 512 long responses. It does not make an optimizer update proportionally smaller.

## Learner evidence

Repacking with the pinned official `BatchPacker` produces **115 microbatches per rank**, with token counts 736,949 / 738,291 / 736,081 / 736,995. The total is 2,948,316 tokens including prompts. Token balance is close; this does not prove equal compute time. The replay initializes the packer and packs on CPU without invoking training or modifying the archive.

The live launcher's and torchrun parent's environments both set `NCCL_P2P_DISABLE=1` and `NCCL_SHM_DISABLE=1`. These settings disable local peer and shared-memory transports. Those flags alone do not prove TCP: earlier runs used InfiniBand. In this run, `ss -tinp` additionally identifies high-volume TCP connections owned by all four trainer processes, connecting the node's address to itself. Their cumulative sent counters total **2.921 TB** at 11:41:40 UTC, counting senders once, not adding received copies.

A 31.23-second passive sample showed:

- Loopback traffic of 1,604.82 MiB/s in each direction. RX and TX describe the same local traffic and must not be summed as unique throughput.
- External eth3 traffic of only 0.098 MiB/s received and 0.569 MiB/s sent; both InfiniBand data counters unchanged.
- Learner GPUs at 99–100% utilization but 0–3% memory-controller utilization, generally 71–128 W, with one 235 W sample. High GPU utilization alone is not evidence of productive matrix computation.
- Each trainer consumed approximately 62 CPU-seconds over 31.23 seconds. Autograd threads were active; trainer file-I/O counters were unchanged throughout the sample.
- All five main inference replicas were idle after completing their cohort. The historical worker had published its first result and was waiting for a new policy.

The pinned training loop calls backward for every microbatch, with FSDP synchronization enabled. CPU optimizer offload is disabled, so its alternate gradient manager is absent. FP32 reductions and repeated parameter/gradient communication over 115 microbatches are substantial work. Activation checkpointing adds recomputation; the current configuration already disables forward resharding. These observations support communication as the first target, but do not isolate transfer time from synchronization waits or rank imbalance.

## What to test next

Use the existing [bounded learner profiler](../scripts/profile_trainer.py) with the same archived cohort and four-rank GPU placement. Compare the current transport against shared memory and healthy peer paths, first verifying collective correctness. Capture a separate short CUDA trace and untraced timing repetitions; keep the batch, FP32 reductions, GRPO normalization and optimizer schedule fixed. The existing replay defaults to two microbatches per rank, so any promising result must then pass a complete 115-microbatch update including weight publication and checkpointing.

Also test reducing synchronization frequency during gradient accumulation, subject to memory and numerical validation. This is a prospective implementation change, not a validated optimization. Changing precision, removing zero-advantage tokens, shortening responses or skipping updates changes the study and is not part of this recommendation.

Do not assume shared memory will produce a particular multiplier: older model replays showed much smaller gains than standalone collective benchmarks. Do not spread learner ranks across more nodes before resolving this local communication path. Extra inference capacity currently cannot remove the learner delay.

Live operator attachment was unavailable under node `ptrace_scope=2` and `perf_event_paranoid=4`. No security settings were changed. Exact forward/backward/collective/diagnostic percentages therefore remain unmeasured. A credible full-run ETA also needs completed updates, checkpoint cost and measurements after the lag-256 transition; startup alone must not be extrapolated over 1,000 updates.
