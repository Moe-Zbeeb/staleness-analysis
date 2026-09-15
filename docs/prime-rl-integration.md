# PrimeRL integration

## Validated revisions

| Component | Revision |
|---|---|
| PrimeRL | `v0.9.0`, `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1` |
| prime-envs | `26dafdc9582576975ec576f893be7319028daf51` for the recorded runs |
| Python | `3.12` |
| Opik | `2.2.61` |

PrimeRL main and `v0.9.0` have different dependency and CUDA requirements. Reproduce these experiments from the pinned revision rather than substituting current main.

## Recommended external-package installation

```bash
git clone --recurse-submodules https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl
git checkout ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1
git submodule update --init --recursive
uv sync --locked --all-extras
uv pip install --editable /path/to/staleness-analysis/packages/prime-rl-staleness
uv run --no-sync python -c "import exact_math, ppo_loss"
```

Keep the extension outside PrimeRL when running experiments. This leaves the upstream checkout clean, so the launch wrappers can verify the exact PrimeRL commit and detect unintended edits.

## Optional in-tree workspace installation

Copy `packages/prime-rl-staleness` into PrimeRL's `packages/` directory. Add it to the root `pyproject.toml` workspace:

```toml
[tool.uv.workspace]
members = [
    "packages/prime-rl-staleness",
]
```

If an existing members list is present, append the entry without replacing the other workspace members. Then run:

```bash
uv sync --package prime-rl --package prime-rl-staleness
uv run --no-sync python -c "import exact_math, ppo_loss"
```

The PrimeRL process only needs both modules importable in the interpreter that launches `rl`. PrimeRL's custom loss loader resolves the configured Python import path at startup.

## Required RL configuration

```toml
[trainer.loss]
type = "custom"
import_path = "ppo_loss.ppo_clip_loss"

[trainer.loss.kwargs]
clip_eps = 0.2

[orchestrator]
batch_size = 64
group_size = 8
max_off_policy_steps = 2

[orchestrator.algo]
type = "grpo"

[[orchestrator.train.source]]
name = "dapo-math"
env.taskset.id = "exact-math"
env.taskset.dataset_path = "/absolute/path/to/train.jsonl"
env.taskset.benchmark = "dapo_train"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"
```

If the package is not installed, the recorded experiment snapshots instead set `PYTHONPATH` to each experiment's `python/` directory. Do not use both mechanisms with different copies of the modules.

## Package boundaries

- `prime-rl-staleness` belongs in the PrimeRL runtime because PrimeRL imports the custom loss and Verifiers imports `exact_math` from the same environment.
- `prime-opik-observer` belongs in a separate environment because its only responsibility is tailing `metrics.jsonl` and forwarding selected metrics.
- Model weights, datasets, checkpoints, credentials, generated manifests, and job-state files are runtime artifacts and are intentionally absent from this repository.

## Launch contract

The recorded scripts expect:

- a Slurm allocation with all trainer and inference GPUs assigned to one job;
- `CUDA_VISIBLE_DEVICES` preserved from Slurm;
- model and dataset preparation completed before smoke or production training;
- a two-step smoke run before the 1,000-step production run;
- checkpoint and orchestrator progress present together before resuming;
- the PrimeRL checkout at the pinned commit with a clean working tree;
- credentials outside source control with owner-only permissions;
- persistent shared storage for checkpoints and node-local storage for model staging and caches.

The experiment snapshots contain the exact MadryLab paths and topology used for their original runs. Change those values deliberately for another cluster; do not treat visible idle GPUs as allocated resources.

