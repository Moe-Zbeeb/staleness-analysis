# Staleness analysis for PrimeRL

This repository preserves the PrimeRL extensions and complete experiment configurations used to study asynchronous GRPO training with bounded rollout staleness. It contains the exact PPO-clipped objective, strict math verifier, four DAPO experiment snapshots, validation and Slurm launch tooling, checkpoint recovery, and isolated Opik telemetry.

The repository does not vendor PrimeRL. The validated runtime is pinned to PrimeRL `v0.9.0` at commit `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1`.

## Contents

```text
packages/prime-rl-staleness/  Installable custom loss and exact-math taskset
packages/prime-opik-observer/ Isolated Opik 2.2.61 telemetry environment
experiments/                  Four complete source/configuration snapshots
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

