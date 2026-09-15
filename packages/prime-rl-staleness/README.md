# prime-rl-staleness

This package installs the two top-level modules expected by the experiment configs:

- `ppo_loss`: exact PPO ratio clipping for PrimeRL's custom RL-loss interface.
- `exact_math`: the strict terminal-answer taskset loaded by `env.taskset.id = "exact-math"`.

Install it into the same Python 3.12 environment as the pinned PrimeRL runtime:

```bash
uv pip install --editable /path/to/staleness-analysis/packages/prime-rl-staleness
uv run --no-sync python -c "import exact_math, ppo_loss"
```

The validated target is PrimeRL `v0.9.0` at commit `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1`.

