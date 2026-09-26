# Corrected fresh exact-256 run

The user authorized stopping job `2144962`, fixing the reviewed issues, and relaunching. The old run was cancelled with no completed optimizer update recorded. This replacement starts from the original DeepSeek 14B weights and a fresh output directory, without reusing old rollout rewards.

The configuration is [exact256-80gb-seed42-v2.json](../../configs/exact256-80gb-seed42-v2.json). Model, prompts, sampling, optimizer, clipped GRPO loss and exact-age schedule remain unchanged: 64 prompts × 8 responses, prompt/response caps 2,048/8,192, seed 42, 1,000 updates comprising 256 on-policy bootstrap and 744 exact-256 updates. Weight decay, extra KL/entropy loss and token filtering remain disabled. Evaluation stays offline; checkpoints remain every 100 updates and at completion on NFS.

## Corrections

| Issue | Implemented change | Validation and limit |
| --- | --- | --- |
| Shared 30-minute deadline | Separate 4-hour learner-publication, 30-minute transfer and 2-hour checkpoint deadlines; upstream collective/publication timeout covers legitimate generation waits | Async tests distinguish learner and transfer delays. Full-size update throughput still needs measurement |
| Archive ordering | Format 2 saves response/question IDs in sample order, preserving upstream sample construction | Real upstream sink tests cover interleaved 8- and 512-response cohorts, including advantages |
| Grading false positives | Structure guard and case-preserving verification; new reward identity and manifest | Targeted regressions and full dataset preparation pass; no claim of v3.2 parity or an overall error rate |
| Recovery memory copies | Preflight returns only completed step; streamed pickle/hash I/O with atomic completion markers | Lifetime, streaming, corruption and recovery tests pass; pending queue remains in CPU memory |

New Runboard fields `study/training_wait_seconds` and `study/weight_transfer_seconds` measure controller phase waits. The former is remaining wait after generation and shipping, not total training compute. PrimeRL's shared broadcast setting also covers generation overlap, so its low-level timeout is 24 hours; actual transfer remains bounded separately.

All changes are in this repository's adapter. Official PrimeRL remains pinned and unmodified. `ProvenanceTrainSink` uses private upstream interfaces documented in the [integration inventory](../upstream-integration.md). See the [grading review](../grading-review.md) for remaining limitations.

## Cluster paths

Paths below are beneath `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study` unless specified otherwise.

- Release: `releases/correctness-v2-20260926`.
- Manifest: `assets/train-manifest-v2.json`, canonical identity `2815cfdbe90623acfbdc2c581b5de9f473f168ee1351d152a6b3b903dec93176`.
- Output: `outputs/exact256-80gb-seed42-v2`.
- Metric mirror: `/mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics/exact256-80gb-seed42-v2`.
- Runboard: [hosted dashboard](https://runboard-cloudflare.mbz02.workers.dev), project `staleness-analysis`.

Job **2144963** was submitted and started on September 26, 2026, at **17:35:51 UTC**, on `deep-chungus-11`. Scheduler and CUDA receipts confirm one exclusive eight-A100-80GB PCIe node, 128 CPU slots, all node memory, account `grad-students`, partition/QoS `high-priority`, 14-day limit and no automatic requeue. Inference uses devices 0–3 and training uses 4–7. The time limit is not a runtime estimate.

The launch directory is `launches/exact256-v2-seed42-20260926T173508Z`; its timestamp records preparation start. It preserves the batch script, frozen configuration, deployment and scheduler receipts, GPU health result, JUnit test report, preflight and Slurm logs. `prepared-runs/exact256-80gb-seed42-v2/SUBMISSION.json` points to it. Source commit is `5ad56dedbd4bfd3a4e3bb2f39ac4e56d0de274d8`; package SHA-256 is `30755021cac742fb1437c437149a1a1013c408fb3945c9d37cdc0af8284442c3`. Later documentation commits do not change the running source snapshot.

All **170 tests passed** on the allocated node in 30.85 seconds, including the CUDA token-mask regression. All eight GPU ranks passed NCCL all-reduce (sum 36), BF16 backward, Flash Attention backward and vLLM RMSNorm checks. Asset preflight accepts 37,703 questions with the corrected manifest identity; its longest retained prompt is 793 tokens.

Runboard registered `d729172f680449f7a9194d0f8c69d799` with name `exact256-80gb-seed42-v2` in project `staleness-analysis`; the hosted API confirms it is running. Source/data/runtime identity is `37a783d0688a010649dc6414860b5d39ca71ea086c6ef11f509923ea0fe222a4`; configuration identity is `cdea779e98127f8a1ded93c8f8d7e61ac1b0b2ea876ac98af929a7ecbd20a60d`.

At **17:41:21 UTC**, startup weight synchronization had completed and the first bootstrap cohort was generating and grading: 86 responses started, 22 completed, and no complete 512-response cohort or optimizer update yet. XFS already contained the configuration, source identity, grading and metric journals. The [timestamped startup evidence](../../diagnostics/runs/exact256-v2-2144963.json) records this observation; it is not a live status report. Full-size update throughput and full-queue memory remain unmeasured.

The preceding readiness run proves small-cohort updates and recovery on the older source. It does not certify this changed source, production-scale training time, full 256-cohort RAM use or checkpoint I/O. New startup results must be recorded separately.

Local validation before deployment: 169 tests passed, one CUDA-only test skipped; Ruff checks and configuration resolution passed. The source-only checks do not substitute for the job's GPU startup checks.
