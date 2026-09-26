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
- Manifest: `assets/train-manifest-v2.json`, SHA-256 `2815cfdbe90623acfbdc2c581b5de9f473f168ee1351d152a6b3b903dec93176`.
- Output: `outputs/exact256-80gb-seed42-v2`.
- Metric mirror: `/mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics/exact256-80gb-seed42-v2`.
- Runboard: [hosted dashboard](https://runboard-cloudflare.mbz02.workers.dev), project `staleness-analysis`.

The launch receipt will record source commit, job ID, allocation, startup checks and Runboard identity. The intended allocation is one exclusive eight-A100-80GB node, 128 CPU slots, all node memory, account `grad-students`, partition/QoS `high-priority`, 14-day requested limit and no automatic requeue. This is a limit, not a runtime estimate.

The preceding readiness run proves small-cohort updates and recovery on the older source. It does not certify this changed source, production-scale training time, full 256-cohort RAM use or checkpoint I/O. New startup results must be recorded separately.

Local validation before deployment: 169 tests passed, one CUDA-only test skipped; Ruff checks and configuration resolution passed. The source-only checks do not substitute for the job's GPU startup checks.
