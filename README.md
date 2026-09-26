# Staleness analysis

A fresh workspace for new Prime RL experiments.

## Previous experiments

The existing work was archived on September 23, 2026 in [prev-experiments/](prev-experiments/README.md), including previously uncommitted source additions.

| Archive | Contents |
| --- | --- |
| [packages](prev-experiments/packages/) | Custom losses, math tasks, and telemetry integration |
| [experiments](prev-experiments/experiments/) | Training code, configurations, launchers, and dependency locks |
| [evaluation](prev-experiments/evaluation/) | Benchmark and evaluation tooling |
| [recoveries](prev-experiments/recoveries/) | Recovery scripts and historical runtime source snapshots |
| [docs](prev-experiments/docs/) | Research notes, integration instructions, and experiment history |

The archive preserves source contents and relative layout. Its recorded paths, job statuses, and setup instructions describe the previous experiments. Reusing a launcher requires reviewing those settings.

Generated job records remain local and ignored. Model weights, datasets, checkpoints, caches, and credentials are excluded from this source archive. Existing Git history is preserved.

## New work

Add new experiment code and configurations at the repository root as their requirements are defined. Keep Prime RL as the training engine and place project-specific tasks, rewards, and losses in separate modules connected through configuration.

## Exact-staleness DeepSeek 14B study

[DeepSeek 14B with cleaned DeepScaleR](experiments/deepseek14b-exact-staleness/README.md) implements one requested exact-k run around pinned official PrimeRL, with delayed rollout cohorts, explicit GRPO settings, isolated grading, checkpoint recovery and staleness auditing. Start with its [review guide](experiments/deepseek14b-exact-staleness/docs/review-guide.md) for the module map, algorithm contract and validation limits. Training has not been launched.

## Prepared math datasets

[Exact-age RL math data](experiments/exact-age-14b-data/README.md) contains the preparation and publication code for independently deduplicated Skywork and DeepScaleR datasets and their cross-deduplicated merge. Only parser-compatible references are published on Hugging Face.

## Reasoning model evaluation

[Five-model reasoning budget evaluation](experiments/reasoning-budget-eval/README.md) contains the 256-question-per-dataset evaluation on cleaned Skywork and DeepScaleR, with 4K/8K/12K generation budgets, native tokenizers, strict final-answer grading, manual-review support, and resumable full-node Slurm execution.
