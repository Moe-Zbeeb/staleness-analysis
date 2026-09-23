# Previous staleness experiments: retained record

Recorded 2026-09-23. User requested deletion of previous experiment checkpoints and logs, with this Markdown account retained. Cleanup status is recorded at the end; historical results below were verified before deletion.

## What was tested

The study tested maximum allowed rollout ages, not exact policy lags. Qwen2.5-Math-1.5B and Qwen2.5-3B each completed 1,000 optimizer updates at caps 2, 4, 6, and 8. These were single-seed runs. Additional 7B and 14B experiment history is recorded separately below when verified; the eight-run numerical analysis must not be extrapolated to those models.

PrimeRL was pinned to v0.9.0, commit `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1`. The training set contained 17,005 prepared DAPO math questions, shuffled during preparation with seed 42 and subsequently traversed in saved order. Each question requested eight answers. Batch selection took 64 traces before zero-advantage filtering. Training used AdamW, learning rate 1e-6, weight decay 0.01, gradient norm limit 1.0, 30 warm-up updates, no learning-rate decay, temperature 1.0, context length 4,096 and maximum completion length 3,072.

Advantages were reward minus the question-group mean, without division by reward standard deviation. The custom token-level PPO surrogate used importance ratios from trainer and generation log-probabilities and clipping bounds [0.8, 1.2]. Global action-token normalization was used. There was no reference KL penalty, entropy bonus, critic or value loss. This custom objective bypassed the framework default loss.

## How the pipeline behaved

Questions were dispatched asynchronously. Completed groups were scored and admitted to a pending training buffer. The trainer selected available traces in insertion order, removed selected traces, and filtered zero-advantage samples. Selected rollouts were consumed once rather than replayed across PPO epochs.

Logged age was `(training_step - 1) - policy.start`. The cap rejected ages above its bound; it did not make younger samples wait. Weight synchronization used pause/keep, updated weights, then resumed unfinished generation with its existing state. Consequently, a trajectory could span multiple policy versions. The starting-version age does not imply that every token was generated under that starting policy. Synchronization did not reset the recorded starting age.

`TARGET_LAG=1` separately limited dispatch/batch lead relative to inference synchronization. It was not an age <= 1 constraint for already-running or buffered work. Age could accumulate during generation and after completion. The retained age histograms alone do not establish which component dominated or whether most trajectories ended synchronized with the trainer.

## Verified eight-run results

Mean age and percentage at the cap are weighted by effective training trace count, not by steps or tokens. Reward and entropy below are averages over updates 901-1000. Training reward describes the effective training cohort after filtering and is not held-out accuracy.

| Model | Cap | Updates | Effective traces | Mean age | At cap | Final-100 training reward | Final-100 entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5B | 2 | 1000 | 27,713 | 1.6230 | 64.15% | 0.3238 | 0.2942 |
| 1.5B | 4 | 1000 | 26,130 | 2.8343 | 31.54% | 0.3047 | 0.2822 |
| 1.5B | 6 | 1000 | 26,117 | 3.2753 | 2.10% | 0.3129 | 0.3119 |
| 1.5B | 8 | 1000 | 27,052 | 3.0336 | 0.00% | 0.3383 | 0.2378 |
| 3B | 2 | 1000 | 23,456 | 1.6231 | 66.51% | 0.3183 | 0.3562 |
| 3B | 4 | 1000 | 24,029 | 2.4077 | 18.01% | 0.3373 | 0.2119 |
| 3B | 6 | 1000 | 23,837 | 2.8451 | 1.28% | 0.3400 | 0.1743 |
| 3B | 8 | 1000 | 22,250 | 3.5329 | 2.05% | 0.3069 | 0.3566 |

The 1.5B cap-8 run contained no effective age-8 traces. The 3B cap-8 run contained 456 effective age-8 traces (2.05%). The 1.5B mean age at cap 8 was lower than at cap 6. These observations demonstrate substantial overlap between the treatments' actual age distributions.

The four-metric training figure did not exhibit the same collapse pattern as Figure 2 of BAPO. This supports stability under the particular observed mixtures, not general robustness to deliberately old data. Differences in models, data, reward/filtering, objectives, update schedules and staleness definitions prevent a direct reproduction claim. Entropy decline alone is not evidence of collapse.

The existing `clip_fraction` counted all sampled ratios outside [0.8, 1.2]. Actual suppression of the clipped policy-gradient term is advantage-aware: positive advantage with ratio > 1.2, or negative advantage with ratio < 0.8. Thus that logged fraction must not be relabeled as the exact fraction of tokens contributing zero policy gradient.

## Interpretation and next study

These were completed cap experiments with a limitation in treatment design, rather than eight failed training runs. Increasing a maximum age did not enforce the intended separation in actual ages. Partial generation across weight updates introduced a further distinction between rollout starting age and token-level behavior-policy mismatch. Neither mechanism proves why collapse was absent.

The next proposed experiment is exact lag k in {2,4,6,8}: generate a complete question group under frozen policy pi_(t-k), preserve original behavior log-probabilities, and train pi_t with one optimizer update per batch. Every admitted sample must satisfy the intended lag. Preserve whole question groups and use each rollout once. A common warm-up and retained consecutive policy versions are required; historical checkpoints spaced 25 updates apart cannot supply this history.

After validating exact ages, compare fixed age 4 against a 50/50 age-2/age-6 mixture, and fixed age 6 against a 50/50 age-4/age-8 mixture. These comparisons hold mean assigned age constant. Record post-filter rollout and token proportions instead of assuming the assigned mix survived filtering unchanged.

Compare the existing clipped loss with selected stale-data corrections only after the age treatment works. Candidate methods include M2PO, BAPO and DPPO; VESPO is a sequence-level extension. Keep reward, normalization, sampling and evaluation matched when isolating loss changes. Start with the existing DAPO preparation for continuity and add DeepScaleR as a separate dataset replication.

Measure held-out accuracy, exact lag, sampled log-ratio tails, squared log-ratio, advantage-aware clipping by age, response length, truncation, raw/effective reward, and generated/retained token budgets. Use repeated seeds for conclusions. Do not change learning rate simply to manufacture collapse.

References: [M2PO](https://arxiv.org/pdf/2510.01161), [BAPO Figure 2](https://arxiv.org/pdf/2510.18927v1), [AReaL](https://arxiv.org/pdf/2505.24298), [DPPO](https://arxiv.org/abs/2602.04879), [VESPO](https://arxiv.org/abs/2602.10693).

## Preserved figures and methodology

Training figures use a centered 25-update moving mean for reward, entropy and clip fraction, with raw gradient norm; an unsmoothed version was also produced. Repeated source records were resolved by taking the last record for each metric and update. Each of the eight runs supplied all 1,000 updates for all four plotted metrics. Rollout-age distributions sum to 100% separately for each run and use effective trace counts.

Derived figures and experiment source/configuration records remain useful historical documentation. They are not substitute checkpoints and cannot resume training after deletion.

## Additional run history

The 7B cap-2 run reached step 1,000 with a final checkpoint. The 7B cap-4 recovery completed steps 1-1,000; its retained completion verification reports successful final evaluation with zero evaluation errors. The 7B cap-6 recovery also reached step 1,000; the fresh inventory verified final trainer metadata and orchestrator progress files. Its parent run had stopped at logged step 277 before continuation. These recovery histories must not be treated as uninterrupted, identical continuations.

The 7B cap-8 original run reached logged step 108, with its latest complete checkpoint at step 100. The 14B original run reached logged step 693, with its latest complete checkpoint at step 675. Additional recovery attempts did not establish completion of either run. At cleanup inventory time, jobs 2142360 and 2142363 were pending to resume those two experiments. Their cancellation and deletion outcomes are recorded below.

Earlier smoke tests and retries included attempts that did not reach an optimizer update. These operational failures are distinct from the completed cap experiments and provide no evidence of staleness-induced training collapse.

## Cleanup status

**Completed and verified on 2026-09-23.** All 703 inventoried cluster target paths were verified absent. A fresh recursive check of the experiment outputs, experiment packages and recovery folders found no remaining checkpoint directories, broadcasts, rollout directories, training metrics or raw log files matching the cleanup criteria. No exported trained-weight shards remained in the experiment release directories.

The deletion receipts account for 30,789 cluster files across those paths. Local cleanup removed 68 copied logs, extracted source-log JSON files, checkpoint metadata files and archives containing checkpoint copies; all 68 were verified absent. Directory targets can contain many files, so target-path and file counts are different measures.

Jobs 2142360 and 2142363 were cancelled before cleanup; the final user job queue was empty. The first SSH connection ended partway through deletion. A fresh connection reconciled the receipts and resumed only remaining targets; final verification resolved that earlier partial status. The resumed cleanup completed without reported deletion errors.

Retained: this Markdown record, training and rollout-age figures, derived plotting tables, experiment code/configuration, dataset and base-model files, completion/evaluation summaries, and cleanup manifests/receipts. Unrelated installation and system diagnostic logs were outside this experiment cleanup. Published model repositories and external tracking services were not modified.

The record is saved locally at `staleness-analysis/docs/previous-experiments-retrospective-20260923.md` and on the cluster at `/mnt/nfs/home/mohamadzbib/projects/rl-infra/reports/previous-experiments-cleanup-20260923/previous-experiments-retrospective-20260923.md`. The corresponding local audit directory is `output/previous-experiments-cleanup-20260923/`; its `resume/complete-manifest.json` and `resume/final-verification.json` provide the final inventory and verification evidence. The old experiments cannot resume from the deleted filesystem checkpoints.
