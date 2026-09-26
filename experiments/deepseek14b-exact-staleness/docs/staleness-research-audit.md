# Staleness tolerance: implementation and research audit

Reviewed 2026-09-26 against study commit `039936f3c5938b8ef9c4ee8b52fae45051ece43f` and official PrimeRL `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1`. The subject is DeepSeek-R1-Distill-Qwen-14B on the pinned cleaned DeepScaleR release. Each requested run has one exact integer lag; this document does not schedule a sweep.

Source links were updated for the subsequent module organization, which preserved scientific behavior; see the [layout and integration notes](upstream-integration.md#upgrades-and-layout-migration). A later baseline change on 2026-09-26 disabled weight decay (`0.01` to `0.0`) to remove parameter shrinkage independent of the GRPO loss gradient. The baseline table below reflects that change. Exact-age scheduling and the loss are unchanged; earlier cluster receipts describe the previous recipe.

**The key distinction is between rollout age, policy difference, and learning quality.** Exact age makes an experiment interpretable, but does not establish that the model tolerates that age. No training or model generation was started for this audit, and there is no measured maximum tolerable lag for this model yet.

The scan traced the authored configuration, queue, controller, loss, data preparation, grader, launcher, provenance and recovery paths; the pinned PrimeRL trainer, loss reduction, optimizer, scheduler, metrics, weight watcher and transport; and primary literature and released M2PO code. This is a source and contract audit, not a proof of distributed execution.

## 1. What we should measure

Let `theta_t` denote the learner after `t` completed optimizer updates. The next update uses behavior policy `mu = theta_(t-k)` after bootstrap.

| Quantity | Meaning | What it cannot establish alone |
| --- | --- | --- |
| Age `k = t - behavior_version` | Number of intervening optimizer updates | How much the probability distribution changed |
| Policy difference | Current versus behavior probabilities on the saved tokens and prefixes | Current held-out problem-solving quality |
| Tolerance | Quality and stability relative to a defined lag-0 reference, under a declared budget and acceptable degradation | A universal safe lag for other recipes or models |
| Efficiency | Time, GPU-hours and generated/trained tokens needed to reach a quality target | Statistical tolerance independent of implementation |

For a sampled token, define `z = log pi_t(token | prefix) - log mu(token | prefix)` and `r = exp(z)`. These include both policy aging and differences between inference and trainer numerics. Report age and distribution diagnostics together. Do not rename a probability-gap metric “staleness” without specifying its units.

Small learning rates or nearly inactive gradients can make an old policy remain close to the learner. A run that barely learns can therefore look stable at a large `k`. Conversely, a large effective update can make a small `k` difficult. These are mechanisms to test, not a measured ranking for this model.

## 2. Findings that affect the study now

### A. The exact-age contract is internally consistent in the traced paths

The queue selects by behavior version, retains original tokens/log-probabilities/advantages, rejects response reuse, and drains the tail without unused cohorts. Generation finishes before applying new inference weights. The controller checks version spans and prompt assignments. The official trainer accumulates packed microbatches, then calls `optimizer.step()` once. Its global token denominator is reduced across ranks, and FSDP averaging is compensated. Microbatch count is not the age clock. [Queue][queue], [controller][controller], [official trainer][trainer], [official loss reduction][reduction].

This establishes the intended software contract. The existing 103 passing tests and GPU component probes do not verify a live 14B weight broadcast, full-length distributed update, or training/resume round trip. A version acknowledgement is not an independent comparison of model contents. Those execution checks remain necessary before interpreting a research run. [Validation record](../diagnostics/cluster-validation.json).

### B. Bootstrap and the learning-rate schedule interact with lag

The first `k` updates use newly generated on-policy cohorts, while distinct cohorts are saved for later consumption. They are labeled bootstrap. With the default 1,000 total updates, `k=32` gives 32 bootstrap and 968 exact-age updates. A larger `k` changes both the warm-start trajectory and treatment duration. It cannot be treated as changing only data age across the entire run. [Queue][queue], [recipe][recipe].

The learning rate warms up for 30 optimizer updates. At `k=32`, the exact-age phase begins after this ramp; a smaller lag can encounter stale data during the ramp. The pinned scheduler implements a nominal zero minimum using a `1e-8` multiplier, making the first update's LR `1e-14` at a `1e-6` peak. The trainer logs LR after `scheduler.step()`, so that logged LR describes the next update, not the update just completed. [Scheduler][scheduler], [trainer][trainer].

Keep the current protocol explicit. A future experiment that introduces lag after a common warm start asks a different causal question and requires an explicit queue-initialization protocol. Do not silently relabel our current runs as that experiment.

### C. PPO clipping leaves an important gradient tail

The implemented objective is the tokenwise clipped surrogate, with centered group rewards and global token averaging:

`L = -(1/N) sum min(r*A, clamp(r, 1-epsilon, 1+epsilon)*A)`.

For positive advantage, the upper side clips. For negative advantage, the lower side clips. **A negative-advantage token with a large ratio remains active.** Norm clipping can limit the resulting total gradient norm, but does not restore information displaced by one dominant contribution. [Study loss][loss].

A CPU-only call to the actual loss, using valid synthetic log-probabilities and `epsilon=0.2`, confirmed these single-token results before global normalization:

| Advantage | Ratio | Loss | Derivative with respect to current log-probability |
| ---: | ---: | ---: | ---: |
| +1 | 2 | -1.2 | 0 |
| -1 | 2 | 2 | 2 |
| -1 | 1000 | 1000 | 1000 |
| -1 | 0.5 | 0.8 | 0 |
| 0 | 2 | 0 | 0 |

The existing outside-range fraction and actual surrogate-clipped fraction correctly measure different things. Add sign-conditioned tails and gradient-contribution diagnostics before drawing a clipping-based explanation of collapse. The finite-ratio guard does not bound large finite ratios or independently certify finite gradients throughout the model.

### D. Training reward is not current-policy evaluation

`updates.jsonl.mean_reward` belongs to the consumed behavior cohort. In the exact-age phase it measures responses generated by `theta_(t-k)`, not fresh responses from `theta_t`. Bootstrap, generation and consumption have different clocks. Episode monitoring is also emitted at generation version, while update receipts use the consumption step. Join by explicit version and response identity. [Controller][controller].

Configuration resolution confirms `orchestrator.eval = null`. The custom controller runs its finite-cohort loop rather than the upstream evaluation loop. Adding an upstream evaluation stanza alone would not establish a working evaluation integration. A frozen held-out evaluator is the largest missing research component.

### E. Age zero still needs numerical calibration

The trainer uses the HF implementation and Flash Attention; inference uses vLLM. Both use BF16 computation, but their kernels, batching and reductions can differ. Quantization and prefix caching are disabled, which removes two confounders without proving equality. [Build configuration][build].

Before comparing lags, score identical token sequences with the same weights in both engines. Check token alignment, EOS, masks, temperature and probability differences, then repeat after weight publication. Thinking Machines demonstrated that trainer/inference numerical differences can materially affect RL; that result motivates calibration here rather than predicting our mismatch magnitude. [Primary experiment](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).

### F. Useful metrics exist, but do not yet support a full mechanism analysis

The trainer aggregates mean, median, standard deviation, minimum and maximum for the custom ratio/log-ratio streams. It also logs entropy, mismatch estimates and gradient norm. Configuration resolution confirms token export is disabled. Saved rollout payloads contain behavior probabilities, not every current learner's probabilities, so full token-level joint analyses cannot generally be reconstructed later from those payloads alone. [Metrics aggregation][metrics], [token exporter][exporter].

Upstream `loss/mean` averages already normalized microbatch loss values; it is not the sum constituting the complete update objective. Upstream token counters include packed/padded input slots, and its sample counter uses microbatch count. Use study response counts and separately measured valid output/loss tokens for scientific budgets. Hardware comparisons should not use these counters interchangeably. [Trainer][trainer].

There is also a concrete analysis-join gap: the controller records response IDs/rewards in episode arrival order, but the upstream sink assembles samples as groups finish. Concurrent groups can interleave arrivals. `TrainingSample` carries no response ID, and the study does not save an explicit sample-index-to-response-ID map. **Do not positionally zip the saved response IDs/rewards with the serialized training samples.** This does not change the advantages attached by the algorithm or the cohort's uniform age, but can corrupt per-response diagnostic attribution. Preserve an explicit mapping before packing when adding research instrumentation. [Controller][controller], [train sink](https://github.com/PrimeIntellect-ai/prime-rl/blob/ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1/src/prime_rl/orchestrator/train_sink.py), [transport fields](https://github.com/PrimeIntellect-ai/prime-rl/blob/ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1/src/prime_rl/transports/batch/types.py).

## 3. Knobs and mechanisms

These are audit findings and hypotheses. “Hold fixed” means fixed within a comparison intended to isolate lag; it does not mean the baseline value has been empirically optimized.

### Age and exposure

| Setting | Current contract | How it affects interpretation |
| --- | --- | --- |
| Lag | Explicit nonnegative integer | Main treatment; exact after bootstrap |
| Update unit | One complete 512-response cohort, one optimizer step | Extra minibatch updates or epochs would change age and reuse |
| Bootstrap and horizon | `k` on-policy updates inside 1,000 total | Changes early learning and number of exact-age updates |
| Model refresh | Every completed update, after generation drains | Skipping refreshes or mixing versions would break the current meaning of `k` |
| Prompt allocation | Consumption-indexed; one seeded shuffle, then cyclic | Preserves assigned questions across lags; not identical sampled responses |
| Admission/replay | Complete groups; no zero-advantage filtering; no replay | Filtering changes the question distribution and effective budget |
| Generation timing | Current weights generate a future cohort; queue delays use | Wall-clock delay does not itself change optimizer-update age |
| Recovery | Pending cohorts preserved; original horizon/layout required | Regenerating pending data or reusing already consumed samples changes the treatment |

Evidence: [queue][queue], [controller][controller], [configuration][config], [recovery](../src/deepseek_study/runtime/checkpoints.py).

### What determines policy movement and sensitivity to old data

| Setting | Current baseline | Mechanism and control |
| --- | --- | --- |
| Initial model/revision | Pinned DeepSeek R1 distilled Qwen 14B | Prior reasoning ability and entropy affect task difficulty and useful signal; conclusions are model-specific |
| Learning rate/schedule | `1e-6`; 30-update ramp; then constant | Changes movement across the lag window; log LR actually used |
| AdamW state | Betas `0.9, 0.999`; epsilon `1e-8`; decay `0` | Momentum and second moments influence movement beyond the current batch; preserve on resume. Decoupled weight decay is disabled |
| Gradient norm cap | `1.0` | Changes update size and direction when rare contributions dominate; measure how often active |
| Prompts per update | 64 | Changes task diversity and gradient variance; keep distinct from response count |
| Responses per prompt | 8 | Changes group-baseline estimation, mixed-success probability and generated-token cost |
| Advantage normalization | Reward minus group mean; no standard-deviation division | Changes prompt weighting and gradient scale; population/sample standardization are supported alternatives, not interchangeable defaults |
| Clipping | Symmetric epsilon `0.2` | Determines which sign/ratio combinations stop contributing; larger or asymmetric intervals change the algorithm |
| Loss reduction | Global valid-token mean | Long responses supply more token terms; total length changes the denominator; not equal weighting per response or a fixed-length denominator |
| Zero-advantage groups | Retained, including in token denominator | Reduces effective signal; an all-zero policy gradient can still be followed by movement from existing Adam momentum |
| Reference KL/entropy bonus | No reference KL; no entropy bonus | No such regularizer currently constrains drift; adding one changes the comparison |
| Training scope | Full model; FP32 optimization/reduction, BF16 compute | LoRA, quantization or different reduction precision would define a different recipe |
| Output length | 8,192 generated-token cap | Changes reachable reasoning, truncation, token weighting and time; track the distribution, not just the cap |

Evidence: [recipe][recipe], [advantages][advantages], [loss][loss], [build][build], [official optimization](https://github.com/PrimeIntellect-ai/prime-rl/blob/ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1/src/prime_rl/trainer/optim/__init__.py).

For independent binary responses with per-prompt success probability `p`, the probability that a group has no reward variation is `p^G + (1-p)^G`. At `G=8` it is about 43% when `p=0.1`, and 0.78% when `p=0.5`. Real responses may be correlated, so measure group outcomes and duplicate completions rather than assuming these values. With one success out of eight, centered advantages are `+0.875` for that response and `-0.125` for the others. Group standardization changes these magnitudes and therefore task weighting.

### Sampling, dataset and reward

| Setting | Current baseline | Mechanism and control |
| --- | --- | --- |
| Temperature and sampling support | Temperature 1; top-k disabled; min-p 0; neutral penalties | The denominator must describe the actual sampling distribution. Altered temperature, nucleus filtering or penalties cannot be treated as an innocuous serving change |
| Top-p | No study override; pinned vLLM default is 1 | Record effective request defaults alongside the resolved study config; model generation defaults are disabled |
| Probability source | Original sampled-token vLLM raw log-probabilities | Recomputing under current or historical trainer weights changes the ratio, especially across backends |
| Prompt/template | Native thinking template plus the fixed boxed-answer instruction | Changes both difficulty and formatting success; BOS/EOS and response masks must remain aligned |
| Prompt/context limits | 2,048 prompt; 10,240 total tokens | Current prepared maximum is 834 prompt tokens; changes to caps/template can change exclusions or truncate supervision |
| Data identity | 37,703 accepted questions from the pinned cleaned release | Cleaning, duplicated content, difficulty mix and train/eval overlap affect apparent tolerance |
| Exclusions | Ten unsupported references; zero oversized prompts | Publish exclusions; do not silently replace rejected questions |
| Reward | Binary correctness; strict complete final box after `</think>` | Formatting failures, false negatives and parser coverage change group advantages |
| Truncated responses | Grade a complete final answer if available | Switching to zero or adding an overlength penalty changes the reward objective |
| Grader deadlines/retries | 8s internal, 10s outer, four workers, one identical-input retry | Distinguish infrastructure failures from wrong answers; do not hide failures as zero reward or replacement sampling |
| Seeds | Seed 42 baseline; seeded data/trainer/inference | Same integer seed does not guarantee paired generations across different schedules or hardware |

Evidence: [taskset](../src/deepseek_deepscaler/__init__.py), [data contract](../src/deepseek_study/dataset/prepare.py), [reward implementation](../src/deepseek_study/dataset/rewards.py), [build][build], [vLLM 0.26 sampling defaults](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/sampling_params.py#L218).

Unique question IDs and a “dedup” dataset name do not establish benchmark decontamination. Audit exact/normalized prompt overlap and available source provenance against the frozen evaluation artifacts before calling results held out.

### Systems controls and observability

| Setting | Current baseline | Research consequence |
| --- | --- | --- |
| GPU layout | 80GB: four trainer/four inference; 40GB: seven/one | Different packing, parallel reductions and generation schedules; keep layout fixed for the main comparison |
| Inference concurrency | 64 requests/16 active sequences per engine on 80GB; 8/2 on 40GB | Intended to change time under the version barrier; can change RNG assignment and numerical batching |
| Attention/compile | Trainer FA2; compile disabled | Changing kernels requires repeating lag-0 probability calibration |
| Checkpointing/offload | Activation checkpointing; activation CPU offload on 40GB; no optimizer offload | Memory/throughput choices; should preserve the mathematical update, but require execution verification |
| Weight transport/cache | NCCL, no transfer quantization, prefix cache off | Verify applied contents and identical-version scoring; count transfer time separately |
| Timeouts | 1,800s synchronization; 86,400s cohort generation | Abort rather than silently admit partial cohorts; compare failure rates, not just surviving runs |
| Checkpoint retention | Every 25; retain last four and every 100th | Plan evaluation/export before checkpoints needed for analysis are pruned |
| Resume RNG | Trainer RNG restored; complete vLLM RNG not restored | Future samples can diverge after resume; annotate restarts and do not claim bitwise equivalence |
| Runtime provenance | Source/dependency/data identities, GPU details | Record driver/CUDA/kernel settings and inherited numerical environment too; a package lock alone does not describe all execution conditions |
| Storage/logging | Pending payloads plus rollout journals and checkpoint bundles | Larger `k` increases resident queue state; storage stalls affect efficiency, not intended age; include checkpoint/evaluation overhead in wall-time comparisons |

Evidence: [recipe][recipe], [launcher](../src/deepseek_study/runtime/launcher.py), [identity](../src/deepseek_study/runtime/identity.py), [trainer RNG](../src/deepseek_study/runtime/trainer_state.py), [weight watcher][watcher], [NCCL transport][nccl].

## 4. Metrics needed for a defensible result

| Evidence | Present now | Needed before the corresponding claim |
| --- | --- | --- |
| Exact age | Per-update min/max, versions, IDs and bootstrap labels | Live broadcast/update verification and an audit joining all resumed segments |
| Current quality | No integrated held-out evaluation | Frozen questions/grader/decoding; evaluate current weights at fixed update ordinals, including initialization and final checkpoint |
| Distribution mismatch | Ratio/log-ratio summaries; `r - log(r) - 1` mismatch statistic | Signed ratio/log-ratio quantiles, log-ratio second moment, extreme-tail mass and valid-token counts |
| Effective learning | Zero-advantage response fraction; gradient norm | All-correct/all-wrong/mixed group counts, informative prompt count, active token fraction, clipping activity, update magnitude |
| Length and reward | Cohort truncation/reward; saved responses and grading reasons | Length quantiles split by correctness/advantage, formatting failures, duplicate response rate and error reasons over time |
| Mechanism attribution | Marginal entropy and ratio summaries | Joint samples of token entropy, ratio, advantage sign, position and length; bounded deterministic sampling with identities |
| Efficiency | Generation/update timing and queue bytes | Valid generated/trained tokens, GPU-hours, fill/steady/drain phases, evaluation and checkpoint overhead, time to a fixed quality target |
| Reliability | Failed episodes/process logs; checkpoint identities | Count every attempted run, resume and infrastructure failure; distinguish interruption from statistical collapse |

Use explicit names for three different quantities: `mean(z^2)`, `mean(r^2)`, and `mean((r-1)^2)`. The M2PO paper's `M2` is the **log-ratio** second moment `mean(z^2)`, not `mean(r^2)`. Its threshold should not be copied onto another statistic. [M2PO §5](https://arxiv.org/html/2510.01161v2#S5).

An empirical weight-concentration diagnostic is `ESS = (sum r)^2 / sum(r^2)`. Specify the unit and report `ESS/N`. Tokens within a response and responses within a prompt are dependent; this is not a count of independent observations or a performance guarantee. Use log-space calculations for large ratios. A sequence product, a geometric mean of token ratios, and a tokenwise ratio define different estimators.

The nonnegative `r - log(r) - 1` statistic has an expected conditional KL interpretation under samples from a normalized behavior distribution with appropriate support. In this study it is measured on old-policy prefixes and includes numerical mismatch. Neither that scalar nor tokenwise importance weighting fully corrects the distribution of prefixes visited by a stale policy. The clipped surrogate should not be described as an unbiased full-trajectory off-policy policy gradient.

## 5. Relationship to the cited experiment

The cited paper uses four optimizer updates per rollout cohort: nominal age 256 spans 256–259. Early updates consume base-model data. It uses standard-deviation-normalized advantages, a reward for answer extraction, constant LR, and models other than our DeepSeek 14B. Its 1,000 training steps must not be equated to our 1,000 optimizer updates. Its results motivate this study but do not validate our chosen lag. [Paper §3 and Appendix B](https://arxiv.org/html/2510.01161v2).

The released Qwen-Math-7B script makes the units concrete: 256 prompts, eight responses each, and a 64-prompt minibatch yield four updates of 512 responses. `stale_iteration=64` delays cohorts. The actor multiplies minibatch size by rollout count. Our 64 × 8 cohort is already one update, so it deliberately avoids the within-cohort age range. [Released script][m2-script], [actor configuration][m2-worker].

The released trainer also recomputes old log-probabilities before enqueueing the cohort; our denominator is the original inference probability. The released PPO loss includes a dual-clip branch, with a very large bound in that script. These are additional implementation differences relevant to extreme tails. The released script specifies `max_steps=1201`, so its checked-in horizon is not identical to the paper's stated horizon either. Treat the publication and this exact code revision as separate evidence. [Released queue][m2-trainer], [released loss][m2-loss].

Call our baseline **centered-advantage, token-mean clipped GRPO at exact optimizer-update lag k**. Neither “paper reproduction” nor “PrimeRL default loss” describes it accurately: pinned PrimeRL defaults use a different DPPO-style loss, while we explicitly supply the custom surrogate. [PrimeRL algorithm documentation][prime-algorithms].

## 6. Research protocol and priorities

1. **Define the claim before selecting results.** Primary question: at a requested `k`, does current-policy held-out quality remain within a predeclared practically meaningful margin of lag 0 under this fixed recipe? Specify budget, evaluation interval, final-checkpoint rule and failure definition. A single run is a pilot, not an estimate of seed variability.
2. **Complete measurement before the main runs.** Implement frozen evaluation and the missing diagnostics; verify generation/consumption/evaluation clocks. Use evaluation RNG and prompts separate from the training stream. If sharing inference hardware, evaluate at a version barrier without altering the queued training cohorts.
3. **Validate execution when training is authorized.** First calibrate same-weight probabilities; then verify a real update, refreshed weights, full-context memory and checkpoint/resume on the chosen layout. No such job is launched by this document.
4. **Hold the scientific recipe fixed for the first comparison.** Model/data/grader, batch/group sizes, LR schedule, normalization, clipping, length/sampling, horizon, bootstrap rule and hardware should match. The user still chooses one run and one `k` at a time; no sweep is scheduled.
5. **Use several independently seeded runs for the eventual claim.** Three seeds is a practical starting proposal, not a universal power calculation. Pair seed labels and prompt assignments where possible; do not assume identical sampled responses. Report seed variation and prompt-clustered evaluation uncertainty, not token-level confidence intervals.
6. **Separate budget questions.** Primary comparison can use the existing equal total-update budget; report the different bootstrap/exact-age exposures. Also report responses, valid tokens and elapsed compute. A secondary equal exact-age-duration design changes total budget and needs separate labeling. Do not mix it into the first comparison.
7. **Investigate mechanisms only after a reference exists.** First examine policy-gap diagnostics and their relationship to quality. Prioritize LR/schedule, clipping/tail behavior, advantage normalization, group size and length/truncation as distinct experiments. Change one factor at a time initially; interactions such as LR × lag or length × lag require later explicit comparisons.

Predeclare collapse as a sustained held-out quality loss or a defined numerical failure, with a window and threshold selected before the run. An infrastructure timeout is a separate outcome. Report the final checkpoint as primary; use a separate validation set for checkpoint selection if desired, and keep final test questions out of that selection. Preserve failed runs rather than dropping them from the analysis.

### Other primary work and how it informs this protocol

| Source | Relevant evidence | Application here |
| --- | --- | --- |
| [Dr. GRPO](https://arxiv.org/abs/2503.20783) | Examines optimization bias and response-length behavior | Declare normalization exactly; centered advantages alone do not identify the complete Dr. GRPO objective |
| [DAPO authors' project](https://dapo-sia.github.io/) | Treats clip asymmetry, dynamic sampling, token-level reduction and overlong rewards as consequential components | Each is a scientific intervention; do not enable them silently while comparing lag |
| [AReaL](https://arxiv.org/abs/2505.24298v5) | Couples asynchronous execution with staleness control and a modified objective | Throughput, age bounds and objective choice need separate accounting |
| [GSPO](https://arxiv.org/abs/2507.18071) | Uses sequence-based importance ratios and clipping | A possible later algorithm comparison, not a replacement with unchanged semantics |
| [verl rollout-correction documentation](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html) | Distinguishes rollout behavior probabilities from the trainer's proximal/reference probabilities | Keep denominator provenance explicit; correction and clipping address different aspects of the mismatch |

These sources identify plausible mechanisms. None provides an empirical tolerance limit for our exact model/data/recipe combination.

## 7. Evidence and limits of this pass

The audit resolved the actual baseline without starting processes: no evaluator, token export disabled, temperature 1, raw behavior probabilities, and unquantized NCCL transfer. The synthetic loss calculations above called the authored loss with one-token inputs; no model was loaded and no optimizer updates were executed. Initial LR was checked using Torch `LinearLR` with the same parameters as the inspected pinned scheduler; importing the complete upstream scheduler in the local CPU environment was blocked by its unavailable `dion` dependency. This does not validate the cluster optimizer through execution.

Previously recorded validation remains 103 tests on macOS and Linux plus cluster component probes. No runtime source or scientific settings were changed during this research scan. The immediate deliverable is this reviewable research specification; the evaluator, richer diagnostics and live end-to-end checks are outstanding work, not completed features.

[queue]: ../src/deepseek_study/rollouts/queue.py
[controller]: ../src/deepseek_study/rollouts/controller.py
[recipe]: ../src/deepseek_study/recipe.py
[config]: ../src/deepseek_study/config.py
[build]: ../src/deepseek_study/runtime/build.py
[advantages]: ../src/deepseek_study/learning/advantages.py
[loss]: ../src/deepseek_study/learning/loss.py
[trainer]: https://github.com/PrimeIntellect-ai/prime-rl/blob/ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1/src/prime_rl/trainer/rl/train.py
[reduction]: https://github.com/PrimeIntellect-ai/prime-rl/blob/ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1/src/prime_rl/trainer/rl/loss.py
[scheduler]: https://github.com/PrimeIntellect-ai/prime-rl/blob/ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1/src/prime_rl/trainer/scheduler.py
[metrics]: https://github.com/PrimeIntellect-ai/prime-rl/blob/ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1/src/prime_rl/trainer/utils.py
[exporter]: https://github.com/PrimeIntellect-ai/prime-rl/blob/ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1/src/prime_rl/trainer/rl/token_export.py
[watcher]: https://github.com/PrimeIntellect-ai/prime-rl/blob/ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1/src/prime_rl/orchestrator/watcher.py
[nccl]: https://github.com/PrimeIntellect-ai/prime-rl/blob/ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1/src/prime_rl/transports/weights/nccl.py
[prime-algorithms]: https://github.com/PrimeIntellect-ai/prime-rl/blob/ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1/docs/algorithms.md
[m2-script]: https://github.com/Infini-AI-Lab/M2PO/blob/af54a3e8feffb7a66a8258003aebefa37759cec0/train-scripts/grpo-qwen-math-7b-s256.sh
[m2-worker]: https://github.com/Infini-AI-Lab/M2PO/blob/af54a3e8feffb7a66a8258003aebefa37759cec0/verl/workers/fsdp_workers.py
[m2-trainer]: https://github.com/Infini-AI-Lab/M2PO/blob/af54a3e8feffb7a66a8258003aebefa37759cec0/verl/trainer/ppo/ray_trainer.py
[m2-loss]: https://github.com/Infini-AI-Lab/M2PO/blob/af54a3e8feffb7a66a8258003aebefa37759cec0/verl/trainer/ppo/core_algos.py
