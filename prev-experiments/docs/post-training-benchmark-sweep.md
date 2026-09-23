# Post-training math benchmark sweep

Requested September 16, 2026 and expanded September 19 to all current/planned staleness caps for the four model families. The original launch gate requires the four production experiments to finish. General reasoning and coding remain outside this math evaluation scope.

## Objective

Measure whether GRPO improves held-out math performance and determine whether the 3,072-token generation cap limits answers. Preserve the current training jobs. The separate evaluation code and pinned configurations are now in [evaluation/README.md](../evaluation/README.md); the first 1.5B family launch is submitted as recorded below.

## September 19 scope and implementation

The user's latest request expands the comparison to every current/planned staleness version: four starting models and caps 2, 4, 6 and 8 for each family, totaling 20 targets. Superseded cap-zero experiments remain outside this scope. The registry currently pins four starting revisions and nine verified final releases; other trained targets remain unavailable until their final exports are verified. This inventory does not replace a live completion check.

The default configuration implements greedy evaluation on 12 sets plus 16-sample competition evaluation and the native 1K/2K/3K comparison. The pinned HMMT February 2026 release contains 33 problems. Full default scope is 560 cells and 194,480 responses before overlap filtering. The five added sets passed a real overlap audit against the exact training rows; full preparation must also read the original frozen retained-evaluation file on the cluster.

Native and Qwen3 thinking-mode code is CPU-tested; allocated-GPU validation remains required. The optional 1.5B 8K extension profiles are planning configurations and are rejected by this native runner until the established v6 context-extension gates are ported and validated. Historical eight-sample results do not satisfy the new sixteen-sample cells. No training or evaluation jobs were changed by this implementation.

### Authorized first launch: completed 1.5B family

**Latest continuation, September 20 at 12:27 UTC:** node1 job **2142048 completed all 60 cells**, saving 15,175 answers. Failed node11 worker 2142053 is superseded by **2142130**, running on two free node11 A10080GB GPUs at **normal priority**. Its corrected native-CUDA assignment passed GPU occupancy and memory checks. It resumes the same remaining 80 cells; startup smoke/generation were pending at this snapshot. Existing report **2142054** now waits for `afterok:2142130` and verifies all 140 cells. Preserve the completed shard and all frozen settings. See the [current continuation and receipt](../evaluation/launches/15b-math-normal-v2-native-20260920/README.md). Earlier job snapshots below are historical.

The user subsequently requested starting the five completed 1.5B versions immediately, before the other families finish. This supersedes the all-four-families waiting condition for this subset. CPU preparation job **2142039** completed all 140 cells. The user then explicitly authorized normal priority on two free GPUs each of deep-chungus-1 and deep-chungus-11. The pending full-node high-priority evaluation **2142040** was cancelled and replaced by normal-priority jobs. The first attempts **2142044/2142045** failed before generation and their report **2142046** was cancelled.

As verified at 18:14 UTC, **2142048** continues running on one GPU on chungus-1. Node11 job **2142049** was cancelled after confirming another user's processes on both assigned GPUs; its pending report **2142050** was also cancelled. Replacement **2142053** is pending at normal priority for two GPUs on chungus-11 with `afterany:2142020:2142041`, conservatively waiting for both existing foreign node11 jobs to finish. Replacement CPU report **2142054** depends on `afterok:2142048:2142053`. All existing preparations, result records and source packages remain preserved. The new wrapper rejects every pre-existing compute process on assigned GPUs and records its separate provenance while preserving the original A10080GB comparison cohort and frozen evaluation settings. No combined benchmark scores are ready.

At the initial launch, node 1 offered only one GPU that was both Slurm-allocated to this evaluation and physically free, because another job used a mismatched GPU ID; the launcher never selects nonallocated GPUs. These partial-node jobs are the current authorized exception to the default scheduling policy. Follow [the launch record](math-sweep-15b-launch-20260919.md) and [the occupancy recovery receipt](math-sweep-15b-occupancy-submission-20260919.json) before creating any later sweep; monitor current jobs **2142048/2142053/2142054** and cell receipts and do not duplicate the 140-cell matrix. The other families retain their readiness gates. This launch covers native contexts only.

### Authorized second launch: completed 3B family

**Current scheduling, September 20 at 13:18 UTC:** the user subsequently requested using only the free GPUs. **2142138** replaces the cancelled full-node job2142137 and requests exactly three GPUs on deep-chungus-5 at normal priority, without exclusivity. It reuses the original validated three-GPU launcher and unchanged preparation2142131; report2142133 now depends on `afterok:2142138`. Preparation continues and GPU startup is unverified. This supersedes the nine-GPU scheduling below. See [the current submission receipt](../evaluation/launches/3b-math-node5-20260920/submission-free-gpus.json).

**Latest expansion, September 20 at 13:09 UTC:** the user authorized all GPUs on deep-chungus-5. The pending three-GPU evaluation 2142132 was cancelled and replaced by **2142137**, requesting the full nine-GPU node exclusively at normal priority. Preparation **2142131** continues unchanged and report **2142133** now waits on `afterok:2142137`. The immutable preparation, 140 cells, models and evaluation settings are preserved. All nine assigned GPUs must pass UUID/PCI, occupancy, memory and BF16 checks before the frozen dispatcher runs. See the [full-node wrapper and submission receipt](../evaluation/launches/3b-math-node5-full-20260920/README.md). GPU startup is not yet verified. At 13:11 UTC, preparation has reached 28/140 cells and a new foreign job 2142136 occupies six node5 GPUs; the nine-GPU worker also needs full-node availability. Earlier snapshots below are historical.

On September 20, the user requested the three free GPUs on deep-chungus-5 for additional evaluations, continuing normal priority. This authorizes the completed Qwen2.5-3B family before the 7B and 14B families finish. Base and final caps 2/4/6/8 are pinned, including the verified cap-6 release. The same audited data and native greedy/sampled protocol produce 140 cells and 48,620 responses.

At 12:55 UTC, CPU preparation **2142131** is running on deep-chungus-6; evaluation **2142132** waits for `afterok:2142131`, requesting exactly three GPUs on deep-chungus-5 at normal priority. CPU report **2142133** waits for `afterok:2142132`. No GPU startup or generation is claimed yet. The launch verifies the original Slurm CUDA selectors, idle GPU occupancy, identical A100 hardware across all workers, host memory, five model smoke tests, decoding and parsing. All comparisons within this family use the same node hardware cohort. At 12:56:58 UTC, preparation is downloading the pinned model snapshot, while Slurm newly reports node5 `MIXED+NOT_RESPONDING`; GPU startup also needs node availability. Preserve this submitted matrix and do not duplicate it. See the [3B launch and receipts](../evaluation/launches/3b-math-node5-20260920/README.md).

## Checkpoints and readiness

Compare each pinned starting model with its final step-1000 checkpoint:

- Qwen/Qwen2.5-Math-1.5B: dapo-qwen25-math15b-grpo
- Qwen/Qwen2.5-3B: dapo-qwen25-3b-grpo
- Qwen/Qwen2.5-Math-7B: dapo-qwen25-math7b-grpo
- Qwen/Qwen3-14B: dapo-qwen3-14b-grpo

Resolve current jobs from live state and provenance. Do not treat earlier archived runs or a successful smoke as completed current production. Verify the final trainer and orchestrator checkpoint, the manifest-bound training completion marker, and clean completion. Wait for all four production runs before launching this sweep. If a run fails, report the concrete failure once and preserve the completed models. Do not restart training, alter scheduling, or substitute archived checkpoints as part of evaluation.

The four starting checkpoints have different pretraining/post-training histories. Report within-model improvement first; do not attribute differences between model families solely to parameter count.

## Benchmark suite

Retain the existing pinned MATH-500, AMC23, AIME24, AIME25, AIME26, Minerva Math and text/final-answer OlympiadBench sets. Add:

- MathArena/hmmt_feb_2025
- MathArena/hmmt_nov_2025
- MathArena/hmmt_feb_2026
- MathArena/brumo_2025
- MathArena/cmimc_2025

Use publisher datasets and pin immutable revisions at preparation time. Check exact, normalized, and near-duplicate overlap with the actual 17,005 training rows before scoring. Exclude confirmed overlaps from the new held-out scores and report exclusions. Do not silently call the filtered result the full official benchmark. A new competition date reduces some overlap risk but does not prove absence of pretraining contamination.

MathArena final-answer datasets expose problem, answer and problem_idx, compatible with conversion to the existing benchmark/prompt/answers/source_id records. Validate fractions, expressions, sets and answer extraction against the publisher grading conventions. Score mathematical final answers; do not describe these results as proof-quality evaluation, including when the original problem asks for a proof. Keep dataset-specific grading differences explicit.

## Evaluation matrix

1. Run every benchmark greedily for both starting and final checkpoints, using the established prompting and 3,072 completion-token budget. Preserve the existing 2,048 cap for Minerva and OlympiadBench in the continuity results. If adding a uniform-cap variant, label it separately.
2. For AIME and the added competition sets, generate 16 independently sampled answers per problem at temperature 0.6, using matched settings and recorded seeds for each starting/final pair. Report sampled mean accuracy, estimated pass@1, pass@4, pass@8 and pass@16. Pass@k is an oracle success metric, not deployed single-answer accuracy.
3. Run a matched 1,024/2,048/3,072 completion-budget comparison on AIME24-26 and HMMT Feb 2026. Count prompt tokens and preserve each model's supported total context. Do not raise the Qwen2.5-Math context above 4,096 as an incidental evaluation change.
4. For Qwen3-14B only, add separate 8,192 and 16,384 completion-budget tests after memory and context validation. Evaluate starting and final checkpoints under identical conditions. Keep non-thinking comparisons separate from an explicitly labeled thinking-mode diagnostic using Qwen's recommended sampled decoding. Do not blend a mode change with the GRPO improvement estimate.

Fix this suite and evaluation matrix before examining new results. Do not select checkpoints, sampling settings or benchmark subsets by the best observed final-test score. Record the actual number of evaluated problems and completions, and all skipped or failed cells.

## Metrics and reporting

Report per-model and per-benchmark baseline/final scores, changes in percentage points, question-level uncertainty intervals, output lengths, truncation, extraction failures, runtime and GPU hours. Resample by problem for uncertainty; repeated completions do not create additional independent questions. Preserve problem identifiers and raw answers so scoring errors are inspectable.

Keep results from current and archived runs distinct. Summarize failures by cutoff, missing/unparseable final answer and mathematically incorrect final answer where evidence permits. A truncated answer does not establish that extra tokens would have made it correct.

## Implementation and execution

Build a separate evaluation-only workflow using the pinned Prime/vLLM runtime and existing exact-math taskset where compatible. Inspect checkpoint format and validate DCP-to-inference export before relying on it. Put any new exported weights under the user's projects/models directory, with persistent NFS backing as needed. Store evaluation inputs, outputs and provenance in a separate post-training-sweep directory.

Validate data membership, grading and checkpoint identity locally or in appropriately allocated CPU work. Run GPU validation and evaluation only inside authorized Slurm allocations. Follow the CSAIL access skill and MadryLab high-priority job policy. Inspect live node topology and QoS limits; use one full compatible GPU node and assign every allocated GPU to inference. Serialize model evaluations when quota requires it. Respect Duo authentication and stop on authentication failure; never retry credentials in a loop.

The follow-up should stay quiet while training is still progressing. Notify only on readiness, a concrete failure, required user action, or completed evaluation results. Avoid duplicate submissions by retaining exact job IDs and completed evaluation cells. Pause the follow-up after delivery.

## Sources checked

- https://github.com/eth-sri/matharena
- https://huggingface.co/datasets/MathArena/hmmt_feb_2025
- https://huggingface.co/datasets/MathArena/hmmt_nov_2025
- https://huggingface.co/datasets/MathArena/hmmt_feb_2026
- https://huggingface.co/datasets/MathArena/brumo_2025
- https://huggingface.co/datasets/MathArena/cmimc_2025
- https://huggingface.co/Qwen/Qwen3-14B
- https://huggingface.co/Qwen/Qwen2.5-Math-7B/blob/main/config.json

## Follow-up state

Before each readiness check, read `tmp/post-training-sweep/state.json` relative to the local workspace when present. It records completed checks, submitted evaluation jobs and previously reported access failures. Respect unresolved SSH retry blocks and suppress unchanged alerts. A successful local control-socket check alone does not prove that forwarding to the destination works.
