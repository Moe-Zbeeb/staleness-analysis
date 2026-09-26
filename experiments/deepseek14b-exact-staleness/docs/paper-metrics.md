# Paper metrics: coverage and measurement contract

The reference papers are [BAPO, arXiv:2510.18927v1](https://arxiv.org/html/2510.18927v1)
and [Prosperity before Collapse / M2PO, arXiv:2510.01161v2](https://arxiv.org/html/2510.01161v2).
Their PDF figures, tables and appendices were inspected on September 26, 2026.
The inventory below covers every experimental figure and table, including repeated
uses of the same observable. Diagrams, notation tables and theoretical functions
are identified separately. This is a measurement integration for our fixed GRPO
experiment; it does not claim to reproduce either paper's training protocol.

## Coverage inventory

Runboard prefixes all new training diagnostics with `paper/`. Existing
`trainer/optim/grad_norm` is the global gradient norm before norm clipping;
`rollout/consumed_reward_mean` is the reward of the cohort being trained on.

| Paper location | Observable | Implementation / destination |
| --- | --- | --- |
| BAPO Fig. 1, 11; Table 1 | AIME 2024 and 2025 accuracy, mean across the two | Offline `evaluation/bapo/*`; average correctness over 16 responses per question |
| BAPO Fig. 2 | Training reward, entropy, clipping fraction, gradient norm | Consumed reward; `entropy/mean`, `clip/fraction`; existing global gradient norm |
| BAPO Fig. 3 | Conceptual algorithm illustration | No additional measured metric; our actual fixed bounds are logged |
| BAPO Fig. 4 | Positive/negative loss contribution and token proportion | `loss/{positive,negative}/absolute_share`, signed `global_token_contribution`, `advantage/*/token_fraction`; zero-advantage tokens counted explicitly |
| BAPO Fig. 5 | Token probability distribution by importance-ratio bin and advantage sign | `joint/{positive,negative}/probability_by_ratio/bin_*`: count, mean, min, quartiles, max; all raw pairs archived |
| BAPO Fig. 6 | Response length split by positive/negative advantage | `length/{positive,negative}/*`; also zero and all responses |
| BAPO Fig. 7 | Reward and entropy under different fixed bounds | Same observed reward/entropy; current `clip/lower_bound`, `clip/upper_bound` |
| BAPO Fig. 8 | Actual lower/upper clipping bounds over time | `clip/lower_bound`, `clip/upper_bound`; constant for our GRPO run |
| BAPO Fig. 9 | Reward, entropy, positive loss share, gradient norm | Metrics above |
| BAPO Fig. 10 | Joint probability, ratio and entropy; entropy 80th percentile | Aligned raw token columns; probability/ratio and entropy/probability bins; `entropy/p80`, high-entropy clipping fractions |
| BAPO Fig. 12–13 | Reward and entropy under partial rollouts / another model | Same observable metrics; those alternate experiments are not enabled |
| BAPO Table 2 | AIME 2024/2025 and MATH accuracy | Offline `evaluation/bapo_llama/*`; MATH is a distinct benchmark from MATH500 |
| BAPO Eq. 6–8, Table 3–4, Fig. 14 | Entropy-change analysis, contribution ratio, notation / theoretical function | Empirical covariance proxy and explicitly named contribution ratios; theoretical per-state covariance and the analytic function are not claimed as measurements |
| M2PO Fig. 1–3, 6, 8, 11; Table 1 | Per-benchmark and mean accuracy; comparisons across methods, staleness, thresholds | Offline `evaluation/m2po/*`; all eight benchmark scores plus equal-weight macro mean |
| M2PO Fig. 1 right, 4a, 7a | Clipping fraction at each optimizer update | `clip/fraction`; outside-range fraction kept separate |
| M2PO Fig. 4b | Mean entropy against absolute distance of ratio from 1 | `joint/entropy_by_ratio_distance/bin_*`; full raw data permit arbitrary pooled update windows |
| M2PO Fig. 5 | Training reward | Consumed-cohort reward |
| M2PO Fig. 7b | Run-average clipping fraction | Both `clip/run_update_mean` and `clip/run_token_weighted_mean`; these have different denominators |
| M2PO Fig. 9 | Squared-log-ratio moment before/after masking and approximate KL | `mismatch/m2`, `mismatch/kl_k1`; `m2_active` and `m2_active_global_denominator` describe our GRPO-active tokens. **M2PO-masked values are not applicable to a GRPO run** |
| M2PO Fig. 10 | Frequencies of clipped token identities | Full per-step ID/count tables in `paper/steps/*.json`, raw IDs in token archives; top-50 ID counts in Runboard |

The eight M2PO benchmarks are AIME 2024, AIME 2025, AMC 2023, AMC 2024,
MATH500, Gaokao, Minerva Math and Olympiad Bench. Additional model/method curves,
threshold ablations and staleness comparisons require separate authorized runs;
logging cannot produce those experiments from one GRPO run. Likewise, BAPO's
adaptive bounds and M2PO's applied mask cannot be reported as if they were active.
No alternate loss, mask, entropy bonus, reference KL, sampling or update rule is
introduced here.

## Tokens with no direct GRPO contribution

The primary metric is **`paper/gradient_signal/noncontributing_token_fraction`**:
the number of valid response tokens whose direct GRPO loss derivative with respect
to their current log probability is zero, divided by all valid response tokens.
Prompt and padding tokens are excluded. Its complement is
`paper/gradient_signal/contributing_token_fraction`; both token counts are logged.

Normally the noncontributing set is the union of zero advantages and the two
clipped directions: `A>0 and r>1+epsilon`, or `A<0 and r<1-epsilon`.
Positive-advantage tokens below the lower bound and negative-advantage tokens
above the upper bound still contribute. `clip/outside_range_fraction` and
`clip/fraction` retain their separate meanings and are not renamed.

The new metric captures a boolean mask on the trainer's device using the same
FP32 ratio arithmetic and PyTorch minimum/clamp derivative conventions as the
loss, including ties, inclusive clamp boundaries and numerical zero coefficients.
The observer counts the full union of rank shards; it does not average rank
fractions. The matching trainer diagnostic is
`trainer/study/noncontributing_token_fraction/mean` (subject to the upstream
trainer's aggregation); use the `paper/` series as the global token fraction.

The disjoint breakdown under `paper/gradient_signal/` is:

- `zero_advantage_fraction`: all zero-advantage response tokens.
- `clipped_zero_gradient_fraction`: nonzero-advantage tokens with zero coefficient in a clipped direction.
- `numerical_zero_fraction`: remaining nonzero-advantage tokens with a zero coefficient.

These fractions add to the noncontributing fraction in the supported unweighted
GRPO baseline. The derivative is measured before global token normalization and
the optimizer. This is a direct per-token loss signal, not a claim that token
embeddings or model parameters receive no gradient through other tokens, nor that
Adam momentum cannot update parameters. No response is dropped or newly masked.

Raw shard schema 2 preserves `surrogate_clipped` and `zero_policy_signal` boolean
columns. `gradient_signal/recorded_mask_fraction` is 1 for this capture path.
Uncached legacy schema-1 raw shards can be analyzed with FP32 CPU reconstruction,
marked 0; this cannot guarantee the original device's rounding at a boundary.
Previously written metric journals and cached summaries keep their historical
values. Use a fresh run/source snapshot for the new logging contract.

## Definitions and important distinctions

All token diagnostics use **valid response tokens only**, excluding prompts and
padding and including zero-advantage responses. For each token, `l` is the current
trainer log probability minus the original inference log probability and `r=exp(l)`.
The denominator remains the exact behavior distribution already used by our loss.
The raw float32 values are preserved; the CPU analysis uses float64 arithmetic.

- `mismatch/m2 = mean(l²)`, with no factor of one half. It is not `mean(r²)`.
- `mismatch/kl_k1 = mean(-l)` follows M2PO Eq. 3; this finite sample estimate can be negative.
- `mismatch/kl_k3 = mean(exp(l)-1-l)` and `kl_absolute = mean(abs(l))` are separately named mismatch statistics. They include trainer/inference numerical differences and stale-context effects; neither is an exact full-distribution KL measurement.
- A token is surrogate-clipped when `A>0 and r>1+epsilon`, or `A<0 and r<1-epsilon`. Merely being outside the interval does not establish clipping of the learning signal. Zero advantages are not counted as clipped.
- `m2_active` averages over tokens with a nonzero direct GRPO coefficient, normally nonzero-advantage, unclipped tokens. `m2_active_global_denominator` puts zeros at excluded tokens but divides by all response tokens. Neither is labelled M2PO masking.
- Let `J=min(r*A,clip(r)*A)`, `P=sum(J[A>0])`, `N=-sum(J[A<0])`. The positive loss share is `P/(P+N)`; the positive-to-negative ratio is `P/N`. Undefined conditional metrics are omitted, with count zero recorded; they are never fabricated as zero.
- BAPO's printed Eq. 8, its figure's contribution shares, and its released recipe are not the same formula. We retain an explicitly named empirical `abs(sum(p_behavior*J[A>0]))/abs(sum(p_behavior*J))` as well as the bounded absolute shares and `P/N`. Near-cancellation can make the former arbitrarily large. It is not used to adjust clipping.
- `gradient_signal/*/absolute_coefficient` is a token-level surrogate gradient coefficient, **not** a positive/negative parameter-gradient norm. Computing those norms would require additional backward passes.
- Entropy is full-vocabulary categorical entropy from the existing trainer forward at behavior-sampled contexts, not sampled surprisal and not behavior-policy entropy. `entropy_clip/empirical_covariance` is the unweighted covariance of sampled current log probabilities and active advantages across this cohort. It is an observational proxy, not the theoretical conditional expectation or a predicted AdamW entropy change.
- The top-entropy cutoff is the observed 80th percentile. Ties are included, so the selected set can exceed 20%.
- Ratio bins use `[lower, upper)`, with a final overflow bin. Probability bins have width 0.05; absolute-ratio-distance bins have width 0.02 through 0.8, then wider tail bins. Raw values preserve arbitrary later rebinning and pooling without binning error.

The [BAPO released recipe](https://github.com/WooooDyy/BAPO/blob/7287044444d5d578529ea464b45f090a36e8da3a/recipe/bapo/policy_loss.py)
and [M2PO released implementation](https://github.com/Infini-AI-Lab/M2PO/blob/af54a3e8feffb7a66a8258003aebefa37759cec0/verl/trainer/ppo/core_algos.py)
were checked for interpretation. In particular, the released M2PO code constructs
bounds from capped log-ratio magnitudes, with additional minimum clip widths;
blindly copying its `M2_after` into a GRPO logger would misrepresent our algorithm.

## Capture, aggregation and clocks

`runtime/trainer.py` temporarily replaces PrimeRL's `setup_token_exporter` factory
with `tracking/tokens.py`. `enable_token_export=true` activates its existing
end-of-microbatch hook and end-of-step flush. Both hooks already exist in the pinned
upstream. The factory is restored in `finally`; upstream source remains unedited.

Each rank collects detached CPU columns from the existing forward, compresses one
NPZ shard per optimizer update, fsyncs it and atomically publishes it. Columns are
token ID, token position, current log probability, original behavior log probability,
advantage and full-vocabulary entropy, plus the two contribution masks and response
lengths. There is no extra model forward/backward or extra tensor all-gather. The study requires one unique DP
shard per trainer rank and rejects context-parallel duplication.

The paper observer concatenates the union of all rank shards. Global means and
quantiles are calculated over that union, not means of rank or microbatch means.
Response-length summaries use response counts. This is CPU work in a separate
process; compression and CPU transfer do add measurable overhead, so wall-clock
throughput comparisons must use identical instrumentation.

Token evidence is captured before the optimizer step. A result is promoted to
`paper-metrics.jsonl` only after the controller's completed-update receipt exists.
The x coordinate `step=33` means update 33 completed; its forward used learner
version 32. Bootstrap is retained and marked in `paper/steps/*.json` and the existing
staleness series. Orphan evidence from an interrupted forward is preserved, but is
not presented as a completed update. Resume segments keep separate run identities;
run-average clipping statistics cover the current segment only.

No positional join between dispatcher arrival order and packed trainer samples is
used. The known different completion orders cannot corrupt these measurements:
advantages and entropy are aligned in the trainer's own packed tensors. Raw IDs
are tokenizer IDs, not question or response identifiers.

## Runboard views

The study emits every finite scalar diagnostic to Runboard under `paper/` and
records binned-chart definitions in run metadata. The updated Runboard dashboard
plots latest-step entropy versus ratio distance, entropy versus probability, and
probability means/quartiles versus ratio, separated by advantage sign. The legend
shows each run's source update. Empty bins cannot carry forward an earlier value.
Time-series smoothing is not applied to relationship plots. Search `bin_` to
inspect the detailed bin counts and histories, which are otherwise hidden to avoid
hundreds of duplicate cards.

These are **binned relationship charts**, not raw 3-D scatter, box-and-whisker plots
or rendered word clouds. Exact joint values and complete clipped-token frequency
tables are saved for those later plots. Top-50 token ID counts are charted only on
updates where that ID is in the top 50; absence is not a zero count. Native dashboard
assets must be deployed to the hosted Runboard service; updating the Python client
alone cannot update an independently hosted Cloudflare dashboard. Older dashboards
still receive and plot all numeric series. The existing hosted dashboard was updated
through [deployment commit 11814e7](https://github.com/Moe-Zbeeb/runboard-cloudflare/commit/11814e7);
its assets, metric pagination and rendered diagnostic relationships were verified.

## Persistent files and XFS mirror

The authoritative files remain under the NFS run directory:

```text
paper/tokens/step_<N>/rank_<R>.npz
paper/tokens/step_<N>/rank_<R>.json
paper/steps/<N>.json
paper-metrics.jsonl
paper-status.json
evaluations/<suite>-step_<N>/{protocol.json,predictions.jsonl,receipt.json}
evaluation-metrics.jsonl
```

The verified fast mount is `/mnt/xfs` (NFS4 export `fast-storage:/fast`), not
`/mnt/xfs1`. The default mirror root is
`/mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics`.
Each run uses its own subdirectory there. Source/run identities, resolved configs,
trainer/study journals, paper diagnostics and imported offline evaluation evidence
are mirrored. Model weights, recovery checkpoints and rollout payloads are not
duplicated. Checkpoints remain on NFS every 100 completed updates plus final.

The launcher verifies mirror ownership and write access before starting services.
Existing data from another run is never overwritten. The observer copies immutable
files atomically, verifies SHA-256 checksums, appends journals from byte offsets,
and validates existing prefixes on replay. `mirror-manifest.json` contains verified
file sizes and checksums after final drain. Temporary mirror outages are retried;
a permanent failure appears in `paper-status.json` and the observer log. There is
no automatic deletion or retention limit for research evidence.

The mount was 98% full (2.8 TB available globally) when inspected; this is not a
personal quota. Six numeric columns and two boolean masks cost 26 bytes per response
token before ZIP compression, at most about 102 GiB for 1,000 updates of 512 responses at 8,192 tokens,
per copy. Real size depends on completion lengths and compression. Do not mistake
this for a guarantee of available capacity over a long run.

The paper observer runs even when `DEEPSEEK_STUDY_RUNBOARD=0`. Network delivery
failure cannot change the GRPO update. Required raw-token file-write failure stops
the trainer rather than silently discarding the requested scientific evidence.
Shutdown grants each observer 20 seconds; interrupted post-processing can be
replayed from the immutable raw archive:

```bash
deepseek-study paper-metrics RUN_DIRECTORY --once
deepseek-study track RUN_DIRECTORY --once
```

Replay of paper metrics is idempotent. A manual `track` invocation creates a new
Runboard import run; it does not append ambiguously to a previous live session.

## Offline benchmark import

Intermediate evaluations stay disabled. The new importer accepts completed,
externally generated/scored predictions for a saved checkpoint; it does not run
inference, download benchmarks, or export PrimeRL's sharded checkpoints to HF.
Those evaluation execution steps remain a separate workflow.

```bash
deepseek-study import-evaluation RUN_DIRECTORY predictions.jsonl protocol.json --step 100
deepseek-study track RUN_DIRECTORY --once
```

Each prediction has `benchmark`, `question_id`, zero-based `repeat`, and boolean
`correct`. The protocol supplies `suite` (`bapo`, `bapo_llama`, or `m2po`),
`grader_identity`, `sampling`, `prompt_template_sha256`, the saved marker's
`checkpoint_components_sha256`, and a `benchmarks` map. Each benchmark entry must
provide all expected `question_ids`, `samples_per_question`, `dataset_revision`
and `dataset_sha256`. BAPO suites require 16 responses per question. The importer
rejects missing/duplicate/extra samples, absent benchmarks and checkpoint identity
mismatches. Accuracy is the mean correctness per question, averaged across
questions, not pass@16. Macro accuracy weights benchmarks equally. Original
predictions/protocols and provenance receipts are retained and mirrored.

The paper does not fully specify every evaluation dataset revision, prompt and
sampling choice. The importer therefore requires explicit provenance instead of
silently inventing a paper-equivalent evaluation protocol. Table 2's `MATH` must
not be relabelled MATH500 without establishing which dataset was used.
