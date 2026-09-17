# Staleness analysis for PrimeRL

This repository preserves the PrimeRL extensions and complete experiment configurations used to study asynchronous GRPO training with bounded rollout staleness. It contains the exact PPO-clipped objective, strict math verifier, four DAPO experiment snapshots, validation and Slurm launch tooling, checkpoint recovery, and isolated Opik telemetry.

The repository does not vendor PrimeRL. The validated runtime is pinned to PrimeRL `v0.9.0` at commit `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1`.

The current study is the [staleness-cap sweep at 4, 6, and 8](docs/staleness-cap-sweep.md), compared with cap 2. The repository preserves each arm's source, configuration, baseline references, and source hashes.

## Experiment status

Last verified September 17, 2026. These are recorded milestones; live progress is available through Slurm and Comet/Opik.

| Experiment | Recorded status | Trainer + inference GPUs |
| --- | --- | --- |
| Original 1.5B / 3B / 7B | Final step-1000 checkpoints verified | See each experiment's provenance |
| Original 14B | Production running; step 151 and paired checkpoint 150 verified | 7 + 2 |
| [1.5B cap 4](experiments/dapo-qwen25-math15b-grpo-stale4-3gpu/README.md) | Completed 1,000 updates | 2 + 1 |
| [1.5B cap 6](experiments/dapo-qwen25-math15b-grpo-stale6-3gpu/README.md) | Smoke passed; production started | 2 + 1 |
| [1.5B cap 8](experiments/dapo-qwen25-math15b-grpo-stale8-5gpu/README.md) | Five-GPU health check passed; smoke started at normal priority | 3 + 2 |
| [3B cap 4](experiments/dapo-qwen25-3b-grpo-stale4-5gpu/README.md) | Submitted and allocated on deep-chungus-7 at normal priority; startup validation in progress | 3 + 2 |

The three-GPU cap-8 snapshot records the superseded setup; the active arm uses five GPUs and separate network ports for concurrent operation with cap 6. Cap 4 and cap 6 use a different topology from the historical 3+2 cap-2 baseline, so staleness is not the only difference in those comparisons. The 3B cap-4 arm also changes topology from its historical 4+3 baseline to 3+2.

The [four-model readiness audit](docs/staleness-sweep-readiness-audit.md) records the earlier feasibility assessment and remaining requirements for extending the sweep to all models. The [zero-staleness tooling](experiments/staleness-zero/README.md) is retained as an unlaunched reference outside the current sweep. Original baseline smoke launchers retain historical allocation settings; consult the readiness audit before reusing them with the final production configurations.

The [post-training benchmark protocol](docs/post-training-benchmark-sweep.md) covers starting-versus-final checkpoint comparisons, additional competition datasets, repeated sampling, and token-budget diagnostics. Evaluation jobs wait for all four original production runs to finish.

## Contents

```text
packages/prime-rl-staleness/  Installable custom loss and exact-math taskset
packages/prime-opik-observer/ Isolated Opik 2.2.61 telemetry environment
experiments/                  Baseline and staleness-arm source/configuration snapshots
docs/objective.md             Exact loss and staleness semantics
docs/prime-rl-integration.md  PrimeRL package and configuration procedure
docs/experiment-matrix.md     Models, topology, and file inventory
```

## Quick start

```bash
git clone --recurse-submodules https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl
git checkout ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1
git submodule update --init --recursive
uv sync --locked --all-extras
uv pip install --editable /path/to/staleness-analysis/packages/prime-rl-staleness
uv run --no-sync python -c "import exact_math, ppo_loss"
```

Select an experiment, replace its cluster-specific absolute paths, prepare its model and data, run its validation, then submit the smoke job before production. The exact package and TOML integration steps are in [docs/prime-rl-integration.md](docs/prime-rl-integration.md).

## Actual training objective

The runs use PrimeRL's GRPO reward assignment and a custom PPO-clipped token loss:

```text
A_i = R_i - mean_j(R_j)
ratio_i,t = exp(log pi_theta(y_i,t) - log mu_i,t)
L = -(1/N) sum_i,t min(ratio_i,t A_i, clip(ratio_i,t, 0.8, 1.2) A_i)
```

The loss uses global action-token normalization. It has no reward-standard-deviation normalization, reference KL penalty, entropy bonus, critic, or value loss. PrimeRL's default DPPO plus KL objective is explicitly bypassed. See [docs/objective.md](docs/objective.md) for the full definition.

## Reproducibility and security

The experiment snapshots retain the original cluster paths and resource topology as provenance. No credentials, model weights, datasets, generated outputs, job-state files, or checkpoints are committed. Runtime credentials must remain in external mode-`600` files.

GitHub updates are deliberate source snapshots. Cluster jobs run independently from their frozen deployed files; a push does not deploy changes or restart training. Models and datasets are published through Hugging Face, while live metrics are sent to Comet/Opik. Generated submission receipts, validation outputs, and mutable job-state files stay outside Git; references to those files in frozen experiment guides refer to the cluster or local audit copy.
