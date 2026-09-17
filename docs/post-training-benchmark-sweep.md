# Post-training math benchmark sweep

Requested September 16, 2026. Default scope: the four GRPO experiments, with mathematical reasoning evaluation after all four finish production training. The user was asked whether to add general reasoning and coding; those domains remain outside this default scope.

## Objective

Measure whether GRPO improves held-out math performance and determine whether the 3,072-token generation cap limits answers. Preserve the current training jobs. This document specifies the planned sweep; an evaluation runner and cluster jobs have not yet been created.

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
