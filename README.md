# Staleness analysis

Research code for studying how rollout staleness affects reinforcement learning. The active training study uses official, pinned PrimeRL and keeps experiment-specific code in its own package.

## Start here: exact-staleness DeepSeek 14B

[DeepSeek 14B on cleaned DeepScaleR](experiments/deepseek14b-exact-staleness/README.md) is the current study. Each invocation prepares one explicit exact lag `k`; it does not schedule a sweep. Training has not been launched.

| What you need | Where to go |
| --- | --- |
| Setup, run commands and current validation status | [Experiment README](experiments/deepseek14b-exact-staleness/README.md) |
| Source layout and execution flow | [Architecture](experiments/deepseek14b-exact-staleness/docs/architecture.md) |
| Baseline hyperparameters | [Run recipe](experiments/deepseek14b-exact-staleness/src/deepseek_study/recipe.py) |
| Allowed settings and validation | [Configuration contract](experiments/deepseek14b-exact-staleness/src/deepseek_study/config.py) |
| What we customize in PrimeRL | [Imported-library integration](experiments/deepseek14b-exact-staleness/docs/upstream-integration.md) |
| Code review and exact-k example | [Review guide](experiments/deepseek14b-exact-staleness/docs/review-guide.md) |
| Research factors, confounders and remaining measurement work | [Research audit](experiments/deepseek14b-exact-staleness/docs/staleness-research-audit.md) |
| Cluster paths, installation and validation evidence | [Cluster guide](experiments/deepseek14b-exact-staleness/docs/cluster.md) |

The code is separated into `learning/`, `rollouts/`, `dataset/` and `runtime/`. Official PrimeRL source files are not edited. There is an explicit process-local checkpoint-factory override, documented with the other integration points in the [dependency guide](experiments/deepseek14b-exact-staleness/docs/upstream-integration.md).

## Related projects

| Project | Purpose |
| --- | --- |
| [Exact-age math data](experiments/exact-age-14b-data/README.md) | Preparation and publication of cleaned Skywork, DeepScaleR and merged datasets |
| [Reasoning budget evaluation](experiments/reasoning-budget-eval/README.md) | Separate five-model evaluation with 4K/8K/12K generation budgets, native tokenizers and final-answer grading |

The separate evaluation project is not automatically integrated into the exact-staleness training study.

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
