# GRPO staleness-cap sweep: 2, 4, 6, 8

Status: design exploration, September 16, 2026. The current direction replaces the proposed cap-0 experiment with caps 4, 6, and 8. No sweep training configurations or jobs have been created. Previously prepared cap-0 files remain as implementation references; they are not part of this sweep.

The subsequent [readiness audit](staleness-sweep-readiness-audit.md) records typed configuration checks, CPU verification, and the remaining launch and validation gaps.

## Experiment matrix

Run each of Qwen2.5-Math-1.5B, Qwen2.5-3B, Qwen2.5-Math-7B, and Qwen3-14B at each new cap. This is **12 new treatment runs**, compared with the four cap-2 baselines. Additional matched cap-2 controls may be needed where hardware differs.

| Arm | Configuration | Allowed rollout starting versions at update 101 |
| --- | --- | --- |
| Existing reference | `max_off_policy_steps = 2` | v98 through v100 |
| New treatment | `max_off_policy_steps = 4` | v96 through v100 |
| New treatment | `max_off_policy_steps = 6` | v94 through v100 |
| New treatment | `max_off_policy_steps = 8` | v92 through v100 |

Age is `(training update - 1) - rollout starting policy version`. These are maximum allowed ages, not required ages. Cap 8 can still train mostly on age-1 or age-2 samples. The experiment measures the effect of allowing older rollouts and the resulting changes in sample survival, throughput, and learning.

## Feasibility in the pinned code

PrimeRL remains pinned to `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1` (v0.9.0). Its configuration already accepts nonnegative caps and defaults to 8. No GRPO loss or framework patch is needed to request 4, 6, or 8.

The train sink drops queued rollouts below `min_fresh_version = (step - 1) - cap` before shipping a batch. The dispatcher cancels live-policy training groups outside that same bound before a weight update. Surviving unfinished generations can keep their prefixes across synchronization, as in the cap-2 baseline. Do not add the cap-0 policy of cancelling every old-policy group to these positive-cap arms.

Keep `TARGET_LAG = 1` unchanged. It limits how far batch submission/new dispatch gets ahead of inference weight updates; it does not impose a maximum age of 1 on existing queued or unfinished rollouts. Increasing it, reducing weight synchronization frequency, increasing concurrency, or injecting delays would introduce additional experimental changes. If the measured age distributions overlap, report that result before considering a separate experiment that explicitly forces lag.

Source locations in the pinned git revision:

- `packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:527`: configurable cap, default 8.
- `src/prime_rl/orchestrator/utils.py:42`: freshness threshold and staleness decomposition.
- `src/prime_rl/orchestrator/train_sink.py:180`: queued rollout freshness sweep.
- `src/prime_rl/orchestrator/dispatcher.py:362`: cancellation before weight synchronization.
- `src/prime_rl/orchestrator/orchestrator.py:97`, `:632`, `:982`: independent fixed lag and dispatch/batch gates.

## Controls to preserve

Each arm starts from the model's original pinned base snapshot with a fresh optimizer, scheduler, task-source state, and observer state. Do not continue a trained cap-2 checkpoint into cap 4, or a cap-4 checkpoint into cap 6. Resumption is allowed only within the same frozen arm.

Hold the following constant per model:

- Model/tokenizer revisions and bytes, prompt template, exact-answer verifier, loss implementation, and all 17,005 prepared training rows and their source ordering.
- 1,000 optimizer updates; batch size 64; group size 8; AdamW learning rate 1e-6, weight decay 0.01, gradient norm limit 1; 30 warmup updates; PPO clip epsilon 0.2; no reference KL penalty.
- Training temperature 1, maximum completion length 3,072, and context length 4,096.
- The existing evaluation datasets, cadence, decoding settings, and 2,048/3,072-token limits. Keep the 14B's non-thinking tokenizer, optimizer CPU offload, and inference TP2 setup.
- Seed settings and concurrency initial/minimum/maximum of 64/16/128.
- Per-model GPU topology, inference parallelism, precision, synchronization, and cache behavior.

Identical seed settings and input ordering do not guarantee identical accepted samples in asynchronous training. Different survival and completion order are consequences of the intervention. Batch filtering and completion lengths can also change the total accepted training tokens, even with equal update counts and token limits; measure that total rather than claiming it is fixed.

## Hardware matching

| Model | Cap-2 provenance | Matching requirement |
| --- | --- | --- |
| 1.5B | Final production job 2140745 started from scratch with 3 trainer + 2 inference GPUs, TP1/DP2 | Preserve 3+2 and match the GPU class; this was a partial-node allocation. |
| 3B | Final production job 2140747 started from scratch with 4 trainer + 3 inference GPUs, TP1/DP3 | Preserve 4+3 and match the GPU class; this was a partial-node allocation. |
| 7B | Used 4+2, then resumed checkpoint 975 on 8+1, TP1/DP1 | Choose one fixed topology for all arms and repeat cap 2 for a strict full-run comparison. The already prepared 8+1 profile is one supported option. |
| 14B | Production profile 7 trainer + 2 inference GPUs, TP2/DP1, optimizer CPU offload | Preserve 7+2, the corrected tokenizer, and the existing execution profile. |

Do not silently resize the small-model runs to fill a larger node and describe staleness as the sole difference. Matching their original partial allocations requires deliberate placement and a live capacity check at submission. If a full-node topology is selected instead, create a fresh cap-2 control with that same topology. The saved inventory is not current availability; no new resource query or submission was performed for this design.

The 7B historical run remains useful context, but it is not a perfect fixed-topology control. Baseline provenance is recorded in `tmp/grpo-audit-20260916/snapshot.json` and `tmp/launch-7b-20260916T100347Z/deployment/runtime-manifest.json` in the local RL infra workspace.

## Evidence that cap 2 is active

The saved metrics snapshot was captured at **2026-09-16 14:58 UTC**. For each production update, the following analysis retains the last stored record with staleness metrics. The 7B file contains 123 repeated update records from restarts; these are not counted as independent updates.

| Model | Updates examined | Mean of per-update mean ages | Updates whose maximum age reached 2 |
| --- | ---: | ---: | ---: |
| 1.5B | 1,000 | 1.624 | 792 / 1,000 (79.2%) |
| 3B | 1,000 | 1.626 | 813 / 1,000 (81.3%) |
| 7B | 1,000 | 1.689 | 898 / 1,000 (89.8%) |

These are batch-level statistics, not the percentage of all rollouts at age 2 or a token-weighted mean. The saved 14B snapshot has only three production updates, so it is excluded from this comparison; that old snapshot does not establish its current completion status. Refresh final run lineage and metrics before the sweep analysis.

## Repository integration

Build a shared cap-sweep preparer around the existing isolated experiment layout. For each model/cap, emit a directory such as `dapo-qwen25-3b-grpo-stale4` and a run name such as `qwen25-3b-grpo-seed42-stale4`. Give each arm its own output, checkpoint link, recovery, observer, and telemetry identity.

Reuse the frozen baseline references, semantic configuration comparison, exact asset copying, GPU mapping, own-run-only resume, and source fingerprints from `experiments/staleness-zero/`. Generate from the verified cap-2 reference, not by successively mutating another treatment arm. Recompute and verify manifests after final changes.

The existing guard cannot be reused unchanged:

- Replace hardcoded cap 0 in static and resolved-runtime validation with the arm's declared cap K. Permit only that cap and operational path/name changes in the production comparison.
- Replace `start == end == t - 1` with raw checks `0 <= (t - 1) - start <= K` and `start <= end <= t - 1`. Mixed-version responses can be valid at positive caps.
- Take `t` from the effective cohort's shipping/update step, not `episode.run.work.step`, which records the task/group's dispatch step.
- Require the logged shipped-cohort maximum to stay within K and retain finite gradient/loss and paired-checkpoint checks. Derive actual age distributions from raw effective traces; aggregate helper metrics clamp negative ages and cannot alone detect future-policy provenance errors.
- Update all copied guard files and their fingerprints together. Keep baseline learning code unchanged.

## Validation and analysis sequence

1. Select and freeze the matching hardware reference for each model. Record any required fresh cap-2 controls.
2. Generate and validate the 12 isolated arms. Confirm that the only production training-setting change is `max_off_policy_steps`.
3. Use a **25-update smoke/validation run**, with the production batch size and generation limit, for each model/cap before full training. A five-update smoke cannot exercise caps 6 or 8. Age 8 first becomes possible at update 9; an age-9 violation first becomes possible at update 10. Longer validation permits exposure but does not guarantee the cap will be reached.
4. Inspect actual age histograms during validation and early production. Technical acceptance requires no bound violations; a useful treatment comparison additionally requires understanding how much the distributions differ. Do not silently alter synchronization to force separation.
5. Start fresh 1,000-update production runs and compare held-out accuracy and training reward at matched update counts. Also plot performance against cumulative accepted tokens and elapsed time. Treat one seed per model as a pilot; uncertainty across seeds requires additional paired seeds, including cap 2.

Track actual age distributions, mean/max and in-flight/queue components; importance-ratio distribution, clip fraction, approximate KL, gradient norm, and reward; truncation and completion lengths; accepted/generated tokens, throughput, discarded work, weight-sync time, and policy-wait time.

Telemetry caveat: `off_policy/dropped` counts queued stale traces, not all cancelled in-flight work. Dispatcher cancellation counters are currently sent through periodic console/W&B logging rather than the file monitor. The Opik bridge also excludes `time/wait_for_policy`, although that metric exists in the raw metrics file. Widening the Opik filter alone cannot recover counters absent from the file. Use existing raw metrics plus console evidence, or apply the same observer-only collection to every arm. Label missing baseline measurements instead of reporting them as zero.
