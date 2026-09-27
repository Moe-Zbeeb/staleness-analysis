# Bounded 1.5B optimization benchmarks

This work follows the [live timing measurements](live-timing-1p5b-20260927.md). Production job `2145184` continues with its frozen configuration while corrected benchmark job `2145240` tests candidates on `deep-chungus-1`. The benchmark has an exclusive eight-A100 allocation, 96 CPUs and a three-hour limit. It uses the normal-priority fallback because an additional full node would exceed the available high-priority GPU quota. These diagnostics are not another study run.

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

The baseline and candidate results were pending when this record was created. No speedup or candidate acceptance is claimed here.

## Production transition requirements

The production inference GPU with UUID `GPU-b03bc3ba-472c-8081-1374-cfb54402f3ac` showed active software thermal throttling. Application batching cannot repair cooling. Moving the job requires healthy replacement hardware and a complete, verified recovery checkpoint containing learner/optimizer state, RNG state, orchestrator progress and the pending exact-staleness queue.

At update 8, no complete checkpoint existed; the first is due at update 100. A restart before that checkpoint would discard completed work. The current recovery contract also rejects changed configuration/source fingerprints. A performance transition must record and validate compatibility explicitly, rather than bypassing these guards or rewriting a checkpoint's provenance. Neither a migration nor a faster production configuration has been applied by these benchmarks.
