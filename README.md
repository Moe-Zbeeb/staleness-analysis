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

## Prepared math datasets

[Exact-age RL math data](experiments/exact-age-14b-data/README.md) contains the preparation and publication code for independently deduplicated Skywork and DeepScaleR datasets and their cross-deduplicated merge. Only parser-compatible references are published on Hugging Face.
