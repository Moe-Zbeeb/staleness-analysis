# Throughput profiling

This investigation uses separate, bounded diagnostic jobs. Production job `2144963`, its source snapshot, queued rollouts, weights and optimizer are not modified. No measured speedup has yet been applied to production. NCCL kernels dominate the traced nine-GPU replay, but enabling shared-memory transport yielded only about 2% observed median improvement in the completed model comparison. No major model-level speedup is established. Raw measurements are in [the results record](../diagnostics/profiling-20260927-results.json).

## Production baseline

Runboard observations at `2026-09-26T21:20:41Z` cover the first three completed updates of `exact256-80gb-seed42-v2`. These are startup observations, not a stable long-run forecast.

| Update | Complete update | Learner forward/backward window | Waiting for the initial cohort | Trainer tokens/s | Reported MFU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 73.09 min | 42.35 min | 30.36 min | 1,193 | 10.9% |
| 2 | 68.98 min | 40.04 min | 28.55 min | 1,186 | 10.8% |
| 3 | 74.88 min | 42.91 min | 31.59 min | 1,193 | 10.9% |

The learner window includes forward/backward, per-token diagnostics, their final archive write, gradient clipping and AdamW. It excludes waiting for generation and weight publication. Consequently, the low reported MFU is not explained solely by the bootstrap wait. Weight transfer is approximately 21 seconds per update; data loading is approximately 0.3 seconds.

The schedule already overlaps each bootstrap learner update with generation of a distinct deferred cohort. The next on-policy bootstrap cohort must wait for the new weights. Reusing answers, generating it from older weights, dropping responses or skipping updates would change the protocol. After 256 bootstrap updates, generation can overlap consumption of the delayed queue.

At these early rates, an illustrative estimate is about 12.9 days for bootstrap and 34.6 days for 1,000 updates, excluding future checkpoint overhead. The Slurm request's 14 days is an allocation limit, not a completion forecast. A full-size update after optimization is needed to replace this estimate.

## Measurements and limits

| Investigation | Evidence | Interpretation |
| --- | --- | --- |
| Token archive CPU/I/O | CPU job `2144978`, `deep-chungus-6`, high priority, completed in 13 seconds; four real rank archives, three repetitions each, exact array round-trip checks | Median compression was 0.803 s and median write/fsync 0.085 s per rank archive. This work is too small to explain a 42-minute update. |
| Production communication | `run.json` selects `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`; topology is PCIe, four learner GPUs in one NUMA domain | Disabling the local transports forces a network path, which can be IB/RoCE or TCP. The original description of this setting as necessarily socket-only was incorrect. |
| Production inference capacity | Each vLLM replica reports 236,048 KV-cache tokens, approximately 23 full 10,240-token sequences; configured maximum is 16 | Modestly higher concurrency deserves a separate benchmark. A cap of 32 could cause preemption on long responses; it is not established as safe or faster. |
| Spare eight-GPU node | Job `2144976` failed NVIDIA topology reporting; independent health job `2144977` failed CUDA initialization with `device=7, num_gpus=7` although Slurm advertised eight GPUs | `deep-chungus-7` is unsuitable for this full-node diagnostic until its GPU exposure is repaired. No model replay ran there. |
| Alternate full-node profile | Job `2144980`, `deep-chungus-4`, nine A100 40GB GPUs, exclusive, account `grad-students`, partition `low-priority`, QoS `normal`, two-hour limit | Health passed on all nine GPUs: collective sum 45, BF16 backward, Flash Attention backward and vLLM normalization. Both transport microbenchmarks passed. Baseline replay completed two updates and then failed with CUDA out of memory on update 3; the alternative model replay never started. Its nine-rank learner layout differs from production and cannot establish a four-rank production speedup. |

The normal-priority GPU allocation uses the standing fallback: production already consumes eight of the twelve GPUs permitted per user at high priority. The diagnostic does not interrupt it to obtain capacity.

## Collected GPU results

NCCL logs show `NET/IB/0` for the baseline and `SHM/direct/direct` for the alternative. The historical files use the label `socket`; retain those filenames as evidence, but interpret them as the network-fallback configuration. New profiling commands use the name `network`.

| Collective | Full tensor | Network median | Shared-memory median | Network time / shared-memory time |
| --- | ---: | ---: | ---: | ---: |
| BF16 all-gather | 64 MiB | 83.21 ms | 134.91 ms | 0.62× |
| BF16 all-gather | 256 MiB | 310.82 ms | 200.79 ms | 1.55× |
| FP32 reduce-scatter | 64 MiB | 80.51 ms | 37.82 ms | 2.13× |
| FP32 reduce-scatter | 256 MiB | 321.02 ms | 208.77 ms | 1.54× |

The rank-zero trace for replay update 2 records 198 all-gather kernels totaling 142.679 seconds and 100 FP32 reduce-scatter kernels totaling 139.468 seconds. These are the GPU kernel entries only; summing their CPU/operator wrapper entries again would double-count them. Communication overlaps some computation, so these totals must not be treated as an additive wall-time breakdown.

The same replay's untraced first update took 287.997 seconds in the learner window. Its traced second update took 305.439 seconds, including profile serialization. Both use two real packed microbatches per rank. NCCL kernels dominate GPU time in this diagnostic. Their duration includes synchronization waits, so this does not isolate physical transfer bandwidth as the cause. All nine traced ranks recorded approximately 282 seconds in NCCL kernels; no single rank had distinctly shorter NCCL time that would identify an obvious GPU-compute straggler. It does not mean a 98% production speedup is available, and the production GPUs have different memory capacity and a different rank layout.

The trace uses NCCL's `RING_LL` kernels despite large model parameter transfers. Follow-up job `2144982` completed all five candidates in 5 minutes 21 seconds, with all correctness checks passing. It compared automatic protocol selection with `NCCL_PROTO=Simple` and eight communication CTAs, including 512 MiB and 1 GiB tensors in independent four- and five-rank groups covering the full nine-GPU node. More CTAs consume additional GPU resources, so a standalone communication improvement must still pass a model replay. See [NVIDIA's NCCL environment-variable documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-min-ctas).

## Four-rank transport matrix

Raw results, health and exact environment settings are in [the transport matrix record](../diagnostics/profiling-20260927-transports.json). These are group-maximum median latencies over six repetitions after three warmups on `deep-chungus-4`.

| Configuration | BF16 all-gather, 512 MiB | FP32 reduce-scatter, 1 GiB |
| --- | ---: | ---: |
| Network, automatic protocol | 519.70 ms | 1,089.56 ms |
| Network, Simple protocol | 521.23 ms | 1,088.14 ms |
| Shared memory, automatic protocol | 113.76 ms | 224.72 ms |
| Shared memory, Simple protocol | 114.75 ms | 225.76 ms |
| Shared memory, Simple, eight CTAs | 114.57 ms | 224.90 ms |

The four-rank group improved by 4.57× and 4.85× with shared memory. Forcing Simple or eight CTAs did not add a useful benefit there. The simultaneous five-rank group improved less: 1.18× for the all-gather and 2.76× for the reduce-scatter with automatic protocol. Both groups ran concurrently and shared host/network resources. Their contention can differ from production, where only one four-rank learner group trains. Placement, group size and concurrency matter; neither result is an end-to-end production speedup. The next model comparison changes only the two local-transport disable flags.

## Corrected model replay

The first replay harness accidentally set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` and `OMP_NUM_THREADS=4`. Production's pinned PrimeRL launcher sets expandable segments to `True` and OMP threads to `1`. The first replay's third update failed trying to allocate 2.90 GiB with 2.62 GiB free and 4.98 GiB reserved but unallocated. Allocator mismatch and profiling retention are possible contributors, not a proven diagnosis of the failure.

The corrected runner imports the same pinned `DEFAULT_COMMON_ENV_VARS` and `DEFAULT_TRAINER_ENV_VARS` as production, records the effective environment per rank, and makes tracing opt-in. Job `2144987` runs an untraced comparison on all nine A100 40GB GPUs on `deep-chungus-4`, with three disposable updates per candidate and a one-hour allocation limit. Account `grad-students`, partition `low-priority`, QoS `normal`, 96 CPUs, all node memory and exclusive allocation were verified. Its health checks passed on every GPU, and both three-update replays completed successfully. Peak memory reached 38.45 GiB after optimizer state allocation and stayed stable. The job completed in 31 minutes 41 seconds.

Weight-transfer job `2144997` completed successfully in 50 seconds; both network and shared-memory probes passed ([receipts](../diagnostics/profiling-20260927-bridge.json)). It assigns four GPUs to inference-side probes and five to trainer-side probes, covering the complete spare node. The broadcast communicator has the same five members as production: one trainer sender and four inference receivers. It tests both transports with each side's real allocator setting. This small-tensor compatibility probe does not load the model or establish the full four-trainer/four-inference production memory fit.

The [complete replay comparison](../diagnostics/profiling-20260927-replay-comparison.json) reports:

| Update | Network | Shared memory | Observed speedup |
| --- | ---: | ---: | ---: |
| 1 | 286.85 s | 282.43 s | 1.016× |
| 2 | 285.18 s | 278.96 s | 1.022× |
| 3 | 284.24 s | 272.79 s | 1.042× |

These sequential, short measurements do not establish a statistically reliable production gain. Packed inputs, behavior log probabilities and advantages matched for all 511,671 token observations across the three updates. Outputs and clipping masks were identical in updates 1 and 2. Update 3 had mean absolute current-log-probability difference 0.008782, maximum 0.248840, and four changed surrogate-clipping/zero-signal decisions among 170,557 tokens. The gradient norm also differed, so this is not bitwise-equivalent training. A repeat with the same transport is needed to distinguish ordinary run-to-run nondeterminism from transport effects; no production change is justified by this comparison alone.

Follow-up job `2145003` isolates one model update with `NCCL_PROTO=Simple`, retaining the same archived microbatches and production allocator/OMP settings. It has a twelve-minute limit on the same full nine-GPU node. The job completed successfully in 5 minutes 52 seconds, including health/startup. Detailed NCCL logs confirm RING/SIMPLE selection and two communication channels. Its learner window was 284.50 seconds versus the baseline first update's 286.85 seconds: 1.008×, less than 1% observed improvement. All 170,557 initial token observations matched exactly, while the gradient norm differed slightly. This single-update test does not establish stability after subsequent updates or a useful throughput gain. See [the Simple-protocol results](../diagnostics/profiling-20260927-simple.json).

The production run, its environment and imported library source are unchanged. Its four-rank A100 80GB topology still requires its own validation before deployment.

## Reproducible tools

| Script | Purpose |
| --- | --- |
| [profile_node.py](../scripts/profile_node.py) | Full-allocation health check, transport microbenchmarks and bounded archive replay; separate output directories; no production checkpoint or policy publication |
| [profile_collectives.py](../scripts/profile_collectives.py) | BF16/FP32 all-gather and reduce-scatter at 16, 64 and 256 MiB, three warmups and twelve repetitions; rank-wise correctness checks and group-maximum latency |
| [profile_trainer.py](../scripts/profile_trainer.py) | Same model/configured GRPO on saved microbatches, phase timers, one optional CPU/CUDA trace and unchanged token diagnostics |
| [compare_profile_replays.py](../scripts/compare_profile_replays.py) | Requires complete optimizer-step receipts and identical grids/configs, compares archived inputs and model outputs, and reports timings, gradient norms and clipping-mask differences |
| [weight_transfer_probe.py](../scripts/weight_transfer_probe.py) | Bounded Torch/vLLM broadcast compatibility test with separate trainer/inference GPU visibility and allocator settings |
| [profile_logging.py](../scripts/profile_logging.py) | Compression and write/fsync timing using actual token archives, with exact column-by-column round-trip verification |
| [profile_transport_matrix.py](../scripts/profile_transport_matrix.py) | Full-node health and bounded network/shared-memory, automatic/Simple protocol and CTA comparisons, with large tensors and explicit rank groups |

Use these tools only inside an allocation. Set both NCCL transport variables before creating any CUDA communicator. The defaults require eight visible GPUs, split into two groups of four; the second round swaps transport placement. `--single-group --expected-gpus 9` instead profiles all nine GPUs sequentially. Every allocated GPU participates. All output directories must be new. `--trace` enables the optional first-round trace; timing comparisons leave it off. `--peer-protocol Simple` and `--peer-ctas N` apply only to the alternative, while the baseline clears protocol/CTA overrides.

The node runner reads the immutable policy-zero warmup archive, repacks its 512 samples with the pinned official `BatchPacker`, and saves the grid. The trainer replays the first two packed microbatches per rank for three disposable optimizer updates, starting from the original base weights for each candidate. The original tokens, masks, behavior log probabilities and advantages are retained. This small repeated sample is a performance diagnostic, not a research training run or a replacement for its full batch.

The fake-data configuration disables the upstream weight sender, but the diagnostic overrides the fake loader with actual archived batches. It disables checkpoint saving and remote monitors in its own process. Phase timing synchronizes CUDA and therefore changes overlap; absolute diagnostic wall time is not an end-to-end throughput measurement. Trace serialization also contaminates the traced update's wall time. Compare phase timings and untraced iterations, then validate any promising option on a full-size update.

The profiler temporarily wraps the imported trainer's forward, loss, optimizer and `Tensor.backward` calls and installs a timed version of our token exporter. These are isolated process-local profiling hooks; the production launcher never imports them. No upstream source file is edited. Restoring or resuming production must use its original frozen release: adding these scripts changes the study source fingerprint because `runtime/identity.py` hashes top-level Python scripts.

## Next decisions depend on results

The tested transport/protocol changes have not established a major model-level speedup. Further work needs an isolated four-rank A100 80GB measurement, a same-configuration numerical repeat, and a full four-trainer/four-inference handoff before deployment. Correct collectives, stable memory and a small-tensor handoff alone do not establish equivalent training or production throughput. Compilation requires its own measured replay and numerical comparison. Selective activation checkpointing is not a simple supported switch for the current Hugging Face Qwen2 implementation; the pinned code supports that mode only for registered layer implementations.

Keep all tokens and optimizer updates. Preserve BF16 compute, FP32 optimizer/reduction, group advantages, global token normalization, exact lag 256, output caps, grading and logging. No shorter answers, zero-gradient-token removal, checkpoint reuse, reduced precision or optimizer offload is an approved speedup from this profile.

## Cluster evidence

Production: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/outputs/exact256-80gb-seed42-v2`.

Initial GPU checks and successful CPU logging measurements: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/profiling/20260927`.

Nine-GPU initial diagnostic: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/profiling/20260927-nine-gpu`.

Completed transport matrix: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/profiling/20260927-transports`.

Corrected replay: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/profiling/20260927-replay-v2`.

Weight-transfer check: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/profiling/20260927-bridge`.

Simple-protocol model diagnostic: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/profiling/20260927-peer-simple`.

User-authorized SSH/Duo reconnection succeeded and collected the initial job's final failure and complete matrix results. The corrected replay, weight-transfer check and Simple-protocol diagnostic all completed separately from production.

Local validation passed Ruff and Python compilation for the profiling tools. The earlier GRPO, queue and paper-metric suites passed 54 tests with one CUDA-only skip; an instrumented AdamW/linear-scheduler smoke check also passed. The replay comparator also passed local smoke checks for equal outputs, modified inputs/outputs/masks and rejection of mismatched grids. The new matrix passed actual GPU correctness checks for all five configurations and both rank groups. These checks do not establish model-level or production-level equivalence.
