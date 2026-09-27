# Live 1.5B timing investigation — September 27, 2026

The running study is generation-limited during bootstrap. Six completed updates average **17m 42s**. A projection that accounts for all three scheduling phases gives **7.20 days of update work**, excluding startup, full recovery-checkpoint pauses and future changes in response lengths. **7–9 days remains a planning range, not a measured completion time or statistical confidence interval.**

Job `2145184` runs `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` on `deep-chungus-7`, with four trainer and three inference A100 PCIe 80GB GPUs. The failed eighth card is excluded. The job, frozen source, model, prompts, rewards, loss, optimizer, GPU assignment and exact-age schedule were left unchanged. This investigation reads existing journals and uses a bounded CPU diagnostic step inside the existing allocation; it submits no model requests or optimizer updates.

Machine-readable measurements are in [the evidence record](../diagnostics/live-timing-1p5b-20260927.json). These observations supersede estimates based on the old 40GB-node timing profile.

## Where the time goes

The table uses controller journals and the trainer's rank-zero metrics for completed updates 1–6.

| Component | Mean | Observed range | Interpretation |
| --- | ---: | ---: | --- |
| Bootstrap cohort generation | 544.10 s | 517.04–568.00 s | Current-policy responses needed before training can start |
| Deferred cohort generation | 514.71 s | 501.90–545.64 s | Future training responses, generated concurrently with the learner |
| Learner work | 379.04 s | 360.80–401.65 s | Inclusive forward/backward, collectives, optimizer, synchronous diagnostics and token export |
| Trainer data loading | 0.384 s | 0.358–0.424 s | Does not include the controller's packing work |
| Controller weight application | 2.192 s | 2.146–2.241 s | The acknowledged weight handoff after generation finishes |
| Remaining controller wall time | 3.041 s | 2.973–3.261 s | Includes weight application; not an additional 3 seconds on top of it |
| Whole bootstrap update | 1,061.85 s | 1,023.53–1,095.48 s | About 17m 42s |

Do not add the learner window to both generation windows. The learner overlaps deferred generation. The observed critical path is approximately:

`544.1 + max(514.7, 379.4) + 3.0 = 1,061.9 seconds`.

The trainer's `time/broadcast_weights` averages 136.09 seconds, but it includes waiting for the inference receiver to finish its cohort. The controller's actual weight-application window is only about 2.2 seconds. Calling the entire 136 seconds a slow network transfer would misdiagnose the run. The controller's extra wait for a published learner version after generation is below 0.5 milliseconds in every measured update.

The trainer's logged throughput is approximately 9,000–9,300 **padded training tokens/s across four GPUs**, not per GPU. Logged model FLOP utilization is about 10.5–10.8%; this is the framework's estimate and does not identify which kernels are responsible. GPU utilization also includes collective kernels and cannot be interpreted as useful matrix-multiply utilization.

## Why model size alone is misleading

The study trains on 64 prompts × 8 responses × 1,000 updates = **512,000 responses**. The first twelve cohorts average **5,965 output tokens per response**, projecting approximately **3.05 billion generated output tokens** if lengths stay similar. These are generated token counts, not padding or input tokens.

There are 1,000 cohorts in total: 256 bootstrap cohorts plus 744 deferred cohorts. All are consumed exactly once. Bootstrap generates two cohorts per update, the middle phase generates one, and the final phase consumes the queue without new generation. The run does not generate an extra 256 unused cohorts.

## Hardware, grading and storage measurements

A 12-minute sample collected 73 observations, ten seconds apart. GPU identities were mapped to the running trainer/inference PIDs through NVML, avoiding the known CUDA/NVML index mismatch on this node.

**Inference GPU `GPU-b03bc3ba-472c-8081-1374-cfb54402f3ac` is thermally throttled.** It stayed at 83–85°C and averaged 817 MHz while reporting 100% GPU utilization. Its peers averaged 1,363 and 1,387 MHz. A subsequent NVIDIA flag query explicitly returned `sw_thermal_slowdown=Active` at 84°C/900 MHz; hardware thermal slowdown, power-cap and power-brake flags were inactive. The other two queried inference GPUs had no active throttle flags at that instant. This establishes throttling, but a controlled before/after measurement is still needed to quantify its end-to-end cost.

| Inference engine | Samples with queued requests | Samples at 16 active sequences | Mean completed-request queue time | Maximum KV-cache occupancy |
| --- | ---: | ---: | ---: | ---: |
| 0 | 16.9% | 22.2% | 1.38 s | 3.44% |
| 1 | 84.8% | 84.2% | 46.00 s | 3.86% |
| 2 | 13.8% | 18.8% | 1.12 s | 3.51% |

These statistics come from 1,287 logged observations per engine, including cohort boundaries and idle intervals. Cumulative completed-request decode times average about 32.3, 45.9 and 31.5 seconds respectively. They reflect different responses, so they are supporting evidence of imbalance, not a controlled per-GPU benchmark. NVIDIA memory reservation near 74 GiB is mostly preallocated KV cache; it does not contradict the low occupied-cache fraction.

Across 7,106 graded responses, grader service latency has mean **3.73 ms**, median **1.39 ms**, p95 **9.91 ms**, maximum **1.57 s** including worker startup. All records have status `ok` and one attempt. Time waiting to acquire a worker is not included in this metric. Grader CPU samples were small; these measurements do not identify grading as the main bottleneck.

The one-second live `iostat` interval showed 0.00% host I/O wait, 88.56% idle CPU, and 0.4%/0% utilization on the two reported NVMe devices. This is a short supporting observation, not a disk bandwidth benchmark. Backup and TensorBoard process CPU costs were small in the longer sample. The backup status remained verified. No full recovery checkpoint existed yet. Trainer and vLLM processes can run on CPUs 0–127 and both NUMA nodes; GPU-local CPU/NUMA binding is another candidate for a controlled test, not a measured speedup.

The all-device topology query timed out and was terminated by its diagnostic timeout; the per-UUID queries of the seven participating GPUs succeeded. The already-known excluded GPU was not used for computation. The host has `ptrace_scope=2`; arbitrary same-user CPU stack attachment is restricted, and no privilege or debugger attachment was attempted.

## Remaining runtime model

| Phase | Updates | Estimated pace | Contribution to full update work |
| --- | ---: | ---: | ---: |
| Bootstrap, updates 1–256 | 256 | 17.70 min/update | 3.15 days |
| Exact age 256 with generation, updates 257–744 | 488 | 8.63 min/update | 2.92 days |
| Exact age 256, queue drain, updates 745–1,000 | 256 | 6.37 min/update | 1.13 days |

Only the bootstrap pace has been measured end to end. Middle/drain timings reuse measured component costs and the inspected queue schedule. At the update-6 receipt, the corresponding remaining estimate is **7.13 days**, before checkpoint pauses. The first complete recovery checkpoint is due at update 100; none has been timed yet. Controller checkpoint sealing happens after its update receipt and outside its recorded `step_wall_seconds`, so future estimates must account for that pause explicitly rather than summing receipts alone.

## Priorities for a faster run

1. **Investigate the slow inference GPU and request imbalance.** Engine 1 accumulates requests much more often than its peers. The hardware measurements and throttle flags above distinguish this from an assumed model-compute limit. Correcting hardware conditions is the first candidate because it does not require shortening responses or changing GRPO.
2. **Benchmark inference batching.** The inherited limit is only 16 active sequences per engine, with 64 total dispatcher requests in flight. Across recorded inference telemetry, maximum KV-cache occupancy is below 4% per engine, and there have been no cache preemptions. Test sequence limits 32 and 64 with compatible dispatcher concurrency on a separate frozen workload; measure complete-cohort throughput and tails, memory, log-probability validity and weight handoff. Larger batches can alter floating-point results and sampled trajectories, so this is not a promise of bitwise-identical rollout data. The memory/throughput tradeoff is documented in the [vLLM tuning guide](https://docs.vllm.ai/en/v0.25.1/configuration/optimization/).
3. **Then benchmark the learner's memory-saving settings.** Peak reserved trainer memory is only about 11.24 GiB of 80 GiB. Full activation checkpointing, FSDP resharding after forward and disabled trainer compilation are inherited conservative settings. Test disabling recomputation, retaining unsharded parameters, or replication on fixed archived batches separately, with gradient/loss/mask and memory checks. PyTorch documents the [resharding memory/communication tradeoff](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html). No speedup for these changes has been measured in this 1.5B run.
4. **Keep the local-storage design and improve backup scaling separately.** No multi-minute packing, transfer or local-write gap is exposed in the measured bootstrap critical path. Background backup currently copies and verifies entire changed files, including growing metric/event journals. That can become expensive later even though it is asynchronous. A chunked or rotated, checksum-verified journal backup is a candidate for a separately tested future change; do not weaken checkpoint verification.

These numerical scenarios show which optimization matters; they are not benchmark results:

| Hypothetical improvement | Projected total update work |
| --- | ---: |
| Current measured component speeds | 7.20 days |
| Generation 1.25× faster; learner unchanged | 5.99 days |
| Generation 2× faster; learner unchanged | 5.23 days |
| Learner 2× faster; generation unchanged | 6.64 days |
| Both 2× faster | 3.62 days |

A learner-only improvement is mostly hidden behind generation until the final queue-draining phase. This is why inference should be investigated first.

## Measurement limits

Generation timing includes dispatch, inference, grading, group completion and local rollout archival. Grader service latencies exclude the wait for an available grader worker and overlap across workers and with decoding. They cannot be summed into generation wall time. Inference request latencies likewise overlap.

The running job has no enabled CUDA operator trace. This read-only investigation does not separately measure attention, MLPs, FSDP collectives, backward recomputation, optimizer kernels or token-export compression. Those need a bounded replay with explicit timers/traces. We have not enabled tracing inside or restarted the production learner. Full checkpoint costs, later response lengths and exact-phase throughput remain unmeasured. No optimization or hypothetical duration above should be described as a validated speedup.
