# Bounded 1.5B optimization benchmarks

Update: the user subsequently authorized a fresh optimized high-priority run. Old production stopped after 13 updates, and diagnostic `2145250` was cancelled. Replacement `2145258` combines bounded validation and conditional full training in one allocation; see the [fresh-run record](runs/deepseek15b-local-v3-opt.md). The matrix results below remain unchanged.

This work follows the [live timing measurements](live-timing-1p5b-20260927.md). Production job `2145184` continues with its frozen configuration while corrected benchmark job `2145240` completed the candidate matrix on `deep-chungus-1` in 23 minutes 51 seconds (exit 0). The matrix records individual candidate failures; its successful scheduler exit does not mean every candidate passed. The benchmark has an exclusive eight-A100 allocation, 96 CPUs and a three-hour limit. It uses the normal-priority fallback because an additional full node would exceed the available high-priority GPU quota. These diagnostics are not another study run.

Initial job `2145232` staged the runtime and passed health checks on all eight A100 PCIe 40 GB GPUs. It was deliberately cancelled when an environment mismatch was found: inference inherited the trainer allocator setting. The corrected harness imports PrimeRL's separate inference defaults, including `expandable_segments:False` and multiprocessing `spawn`, and records the effective inference environment. Job `2145240` reuses the staged files but creates fresh job-specific result directories and repeats the health check and complete matrix. No timing from the cancelled attempt is accepted as an optimization result. The intervening launch `2145239` exited before GPU work because the system Python lacked `hashlib.file_digest`; the cache-reuse wrapper now uses the existing streaming hash helper.

## Preserved experiment

The benchmark stages the production model and tokenizer with SHA256 checks and replays the archived policy-zero cohort. It preserves the model, response limit, sampling temperature, learner precision, optimizer and GRPO loss. It does not edit imported PrimeRL or vLLM files. The installed vLLM `core_client.py` matches its wheel RECORD checksum and already includes load-aware routing and rotating tie-breaking; the queue imbalance alone does not establish a routing defect.

Production retains 1,000 updates, exact lag 256 after bootstrap, 64 prompts with eight responses each, 8,192 response tokens, zero weight decay, checkpoint interval 100 and TensorBoard-only tracking. No production weights, optimizer state, queued rollouts or outputs are modified by these scripts.

## Inference matrix

[`benchmark_inference.py`](../scripts/benchmark_inference.py) evaluates maximum active sequence counts 16, 32 and 64, followed by a repeated 16-sequence baseline. Four independent single-GPU workers each process 128 requests from the same 512-response prompt workload. Each request has a fixed seed. The script checks request coverage, prompt identity, completion bounds and finite behavior log-probabilities, and records per-response token counts and hashes.

Initialization, short warmup and generation are timed separately. Compare both complete-cohort time and output tokens per second: batching may change sampled token sequences and lengths. This test measures single-engine capacity, not the production data-parallel router, dispatcher, grader or weight handoff. Production has three inference GPUs, while the diagnostic uses four. Its aggregate throughput is not a measured production speedup. Dispatcher concurrency candidates 128 and 256 still require a bounded end-to-end validation after a batching candidate passes.

## Learner matrix

[`profile_trainer.py`](../scripts/profile_trainer.py) runs three optimizer updates on eight archived packed micro-batches per rank, using the other four GPUs. Cases are baseline, repeated baseline, activation checkpointing disabled, resharding disabled, both disabled, and compilation enabled. Only `ac`, `reshard_after_forward` and `compile` can be overridden. Each case has a 30-minute deadline and a fresh output directory.

[`compare_profile_replays.py`](../scripts/compare_profile_replays.py) requires identical archived input tensors and explicitly allowlists the execution settings under comparison. Changes to the loss, optimizer, precision or other trainer configuration are rejected. It compares current log-probabilities, entropy, clipping/no-signal masks, losses and gradient norms across all three updates. A repeated baseline establishes the numerical and timing variation of the replay itself. The comparison records differences; it does not automatically approve a candidate. It does not compare complete gradient tensors.

CUDA synchronization used for phase measurements adds overhead. The bounded replay is smaller than a production learner update and does not measure rollout generation, weight handoff or recovery checkpointing. A faster replay must still pass a full-size memory and correctness check before adoption.

## Runtime and receipts

[`benchmark_1p5b.py`](../scripts/benchmark_1p5b.py) stages the frozen source, local Python runtime, model and immutable rollout archive, then performs an eight-GPU health check before starting either matrix. It records GPU memory, temperature, clock and utilization samples. Diagnostic workers use node-local working directories and compilation caches; results are copied to shared storage after each case.

- Cluster controls: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/profiling/optimize-1p5b-20260927/control`
- Job-local workspace: `/tmp/staleness-opt-1p5b-2145232`
- Local receipts: `/tmp/staleness-opt-1p5b-2145232/results-2145240`
- Shared receipts: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/profiling/optimize-1p5b-20260927/results-2145240`

## Collected results

The [complete results receipt](../diagnostics/optimization-1p5b-20260927.json) includes all four inference cases, all completed learner comparisons, source artifact hashes and the rejected out-of-memory case. The earlier [partial receipt](../diagnostics/optimization-1p5b-20260927-partial.json) is retained as a historical snapshot.

| Inference case | Median per-GPU output tokens/s | Slowest shard generation time | Output tokens | Truncated responses |
| --- | ---: | ---: | ---: | ---: |
| 16 sequences | 2,561 | 309.23 s | 3,122,423 | 42.97% |
| 32 sequences | 3,820 | 205.50 s | 3,125,491 | 41.60% |
| 64 sequences | 5,344 | 150.60 s | 3,143,382 | 45.31% |
| 16 sequences repeated | 2,561 | 309.07 s | 3,122,423 | 42.97% |

Every case completed all 512 responses with finite behavior log-probabilities. The 32- and 64-sequence cases delivered 1.49× and 2.09× the baseline median per-GPU throughput. Repeating the baseline reproduced all response token hashes. Increasing batching changed 509 and 510 of the 512 responses respectively, despite fixed request seeds. These are execution settings with observable numerical/sampling effects; they do not guarantee identical sampled trajectories.

| Learner candidate | Median replay speedup | Peak allocated GPU memory | Decision |
| --- | ---: | ---: | --- |
| Repeated baseline | 1.003× | 11.24 GiB | Timing and numerical reference |
| Disable activation checkpointing | 1.023× | Approximately 38 GiB | Keep activation checkpointing enabled |
| Disable forward resharding | 1.258× | 14.15 GiB | Candidate for full-size validation |
| Disable both | Not available | Out of memory on 40 GB GPUs | Rejected |
| Enable compilation | 0.984× | 11.24 GiB | Keep compilation disabled |

The identical learner baseline repeat was not bitwise identical: 21 clipping/no-signal decisions differed across the three updates. The no-checkpointing, no-resharding and compilation cases had 15, 18 and 14 differences respectively, with archived inputs equal. These counts do not define an equivalence tolerance or prove identical learning trajectories. Complete gradient tensors were not compared. The promising resharding change reduced the replay's backward phase; the activation-checkpointing and compilation alternatives showed no useful gain.

These component speedups cannot be multiplied into a production speedup or converted directly into a new completion estimate. Generation overlaps learner work, the matrix has a different inference topology and GPU memory class, and the later exact-lag and draining phases remain unmeasured.

## Full-pipeline validation

[`benchmark_pipeline.py`](../scripts/benchmark_pipeline.py) prepares two isolated cases using the actual dispatcher, grader, learner and weight handoff. Each case has four trainer GPUs, four inference GPUs and exactly two committed bootstrap updates, with a 2,400-second supervisor limit. The production configuration still specifies 1,000 updates and lag 256; a bounded supervisor stops the diagnostic before it can become a full study.

The baseline uses 16 active sequences, dispatcher concurrency 64 and forward resharding enabled. The candidate uses 64 active sequences, concurrency 256 and forward resharding disabled. Both retain activation checkpointing and leave compilation disabled. Scientific settings, archived matrix input checks and dependency integrity checks are preserved. The script rejects adapter changes beyond the two explicitly checked resharding-option edits, verifies staged assets and runs the eight-GPU health probe before training.

Job `2145250` is submitted with an exclusive eight-GPU A100 allocation, normal QoS, a two-hour wall limit and no automatic requeue. At the last check on September 27 it was **pending for resources** on node 1, which holds the verified local diagnostic runtime. Another allocation occupies that node. The full-pipeline candidate therefore remains unvalidated. Scheduler start estimates can change and are not a promised start time.

Controls and results are under `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/profiling/optimize-pipeline-1p5b-20260927/`. Each case writes locally and copies results to its separate shared diagnostic directory. The comparison excludes startup from update timing and must distinguish deliberate bounded shutdown from a crash. This test does not measure recovery-checkpoint cost or the exact-age phase, and its four-inference-GPU topology differs from production's three.

The adapter now exposes `trainer_reshard_after_forward`, defaulting to `true`, and passes it through to the existing official PrimeRL option. Setting it to `false` keeps gathered parameters after forward computation and was the tested resharding candidate. It changes the configuration fingerprint; existing runs must continue with their frozen release until an explicit compatible transition is validated. Tests verify that this switch changes only the corresponding official execution setting. Imported library source is unchanged.

## Production transition requirements

The production inference GPU with UUID `GPU-b03bc3ba-472c-8081-1374-cfb54402f3ac` showed active software thermal throttling. Application batching cannot repair cooling. Moving the job requires healthy replacement hardware and a complete, verified recovery checkpoint containing learner/optimizer state, RNG state, orchestrator progress and the pending exact-staleness queue.

At the last verified production update, 11, no complete checkpoint existed; the first is due at update 100. A restart before that checkpoint would discard completed work. The current recovery contract also rejects changed configuration/source fingerprints. A performance transition must record and validate compatibility explicitly, rather than bypassing these guards or rewriting a checkpoint's provenance. Neither a migration nor a faster production configuration has been applied by these benchmarks.
