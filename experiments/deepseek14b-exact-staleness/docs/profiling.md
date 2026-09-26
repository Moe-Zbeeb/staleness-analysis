# Throughput profiling

This investigation uses separate, bounded diagnostic jobs. Production job `2144963`, its source snapshot, queued rollouts, weights and optimizer are not modified. No measured speedup has yet been applied to production.

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
| Token archive CPU/I/O | CPU job `2144978`, `deep-chungus-6`, high priority, completed in 13 seconds; four real rank archives, three repetitions each, exact array round-trip checks | An observed rank took 0.804 s to compress and 0.100 s to write/fsync approximately 7.8 MB. This work is too small to explain a 42-minute update. GPU-to-CPU export cost still needs the trainer trace. |
| Production communication | `run.json` selects `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`; topology is PCIe, four learner GPUs in one NUMA domain | Socket transport is a concrete candidate. Earlier weight-transfer correctness probes passed with explicit peer/shared-memory transport at startup, but did not measure sustained FSDP performance. |
| Production inference capacity | Each vLLM replica reports 236,048 KV-cache tokens, approximately 23 full 10,240-token sequences; configured maximum is 16 | Modestly higher concurrency deserves a separate benchmark. A cap of 32 could cause preemption on long responses; it is not established as safe or faster. |
| Spare eight-GPU node | Job `2144976` failed NVIDIA topology reporting; independent health job `2144977` failed CUDA initialization with `device=7, num_gpus=7` although Slurm advertised eight GPUs | `deep-chungus-7` is unsuitable for this full-node diagnostic until its GPU exposure is repaired. No model replay ran there. |
| Alternate full-node profile | Job `2144980`, `deep-chungus-4`, nine A100 GPUs, exclusive, account `grad-students`, partition `low-priority`, QoS `normal`, two-hour limit | Submitted and observed running through the health-probe launch. Results and final status still require collection. Its nine-rank learner layout differs from production and cannot establish a four-rank production speedup. |

The normal-priority GPU allocation uses the standing fallback: production already consumes eight of the twelve GPUs permitted per user at high priority. The diagnostic does not interrupt it to obtain capacity.

## Reproducible tools

| Script | Purpose |
| --- | --- |
| [profile_node.py](../scripts/profile_node.py) | Full-allocation health check, transport microbenchmarks and bounded archive replay; separate output directories; no production checkpoint or policy publication |
| [profile_collectives.py](../scripts/profile_collectives.py) | BF16/FP32 all-gather and reduce-scatter at 16, 64 and 256 MiB, three warmups and twelve repetitions; rank-wise correctness checks and group-maximum latency |
| [profile_trainer.py](../scripts/profile_trainer.py) | Same model/configured GRPO on saved microbatches, phase timers, one optional CPU/CUDA trace and unchanged token diagnostics |
| [profile_logging.py](../scripts/profile_logging.py) | Compression and write/fsync timing using actual token archives, with exact column-by-column round-trip verification |

Use these tools only inside an allocation. Set both NCCL transport variables before creating any CUDA communicator. The defaults require eight visible GPUs, split into two groups of four; the second round swaps transport placement. `--single-group --expected-gpus 9` instead profiles all nine GPUs sequentially. Every allocated GPU participates. All output directories must be new.

The node runner reads the immutable policy-zero warmup archive, repacks its 512 samples with the pinned official `BatchPacker`, and saves the grid. The trainer replays the first two packed microbatches per rank for three disposable optimizer updates, starting from the original base weights for each candidate. The original tokens, masks, behavior log probabilities and advantages are retained. This small repeated sample is a performance diagnostic, not a research training run or a replacement for its full batch.

The fake-data configuration disables the upstream weight sender, but the diagnostic overrides the fake loader with actual archived batches. It disables checkpoint saving and remote monitors in its own process. Phase timing synchronizes CUDA and therefore changes overlap; absolute diagnostic wall time is not an end-to-end throughput measurement. Trace serialization also contaminates the traced update's wall time. Compare phase timings and untraced iterations, then validate any promising option on a full-size update.

The profiler temporarily wraps the imported trainer's forward, loss, optimizer and `Tensor.backward` calls and installs a timed version of our token exporter. These are isolated process-local profiling hooks; the production launcher never imports them. No upstream source file is edited. Restoring or resuming production must use its original frozen release: adding these scripts changes the study source fingerprint because `runtime/identity.py` hashes top-level Python scripts.

## Next decisions depend on results

Transport changes need correct collectives, matching replay outputs within justified floating-point tolerances, stable memory, and a successful four-trainer/four-inference handoff. Compilation requires its own measured replay and numerical comparison. Selective activation checkpointing is not a simple supported switch for the current Hugging Face Qwen2 implementation; the pinned code supports that mode only for registered layer implementations.

Keep all tokens and optimizer updates. Preserve BF16 compute, FP32 optimizer/reduction, group advantages, global token normalization, exact lag 256, output caps, grading and logging. No shorter answers, zero-gradient-token removal, checkpoint reuse, reduced precision or optimizer offload is an approved speedup from this profile.

## Cluster evidence

Production: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/outputs/exact256-80gb-seed42-v2`.

Initial GPU checks and successful CPU logging measurements: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/profiling/20260927`.

Nine-GPU diagnostic scripts, job log and results: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/profiling/20260927-nine-gpu`.

The authenticated SSH transport closed after job `2144980` was observed running. One user-authorized reconnect timed out after 180 seconds at the jump-connection stage, before any password or Duo prompt. No further automated SSH attempts were made. The diagnostic's completion, health receipt and GPU timings remain unverified; the Slurm job has a two-hour allocation limit. Production was last observed running with three completed updates.

Local validation passed Ruff and Python compilation for all four profiling tools. The GRPO, queue and paper-metric suites passed 54 tests with one CUDA-only skip; an instrumented AdamW/linear-scheduler smoke check also passed. These checks do not substitute for collecting the GPU job's results.
