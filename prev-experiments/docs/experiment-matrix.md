# Experiment matrix

| Experiment | Model | Trainer GPUs | Inference GPUs | Purpose |
|---|---|---:|---:|---|
| `dapo-qwen25-3b-grpo` | Qwen2.5 3B family | 4 | 2 | Small-model control run |
| `dapo-qwen25-math7b-grpo` | Qwen2.5 Math 7B | 4 | 2 | Primary math scaling run |
| `dapo-qwen25-math15b-grpo` | Qwen2.5 Math 1.5B | 4 | 2 | Small math-model comparison |
| `dapo-qwen3-14b-grpo` | Qwen3 14B | 4 | 2 | Larger post-trained model comparison |

All four snapshots use the same custom PPO-clip implementation, exact terminal-answer verifier, group size `8`, batch size `64`, clip epsilon `0.2`, no reference KL penalty, seed `42`, and `max_off_policy_steps = 2`.

Each experiment directory contains:

- `config/main.toml`: production configuration.
- `config/smoke.toml`: bounded end-to-end validation configuration.
- `config/data_exclusions.json`: audited contamination exclusions.
- `python/`: the exact local modules imported by the recorded run.
- `scripts/prepare_data.py`: pinned datasets, cleaning, decontamination, row ordering, and manifest generation.
- `scripts/validate_experiment.py`: data, tokenizer, reward, loss, and configuration checks.
- `scripts/smoke_job.sh`: health gate and two-step full-stack test.
- `scripts/train_job.sh`: production launch, checkpoint recovery, requeue handling, local staging, and telemetry lifecycle.
- `experiment.json`: model, dataset, revision, topology, and objective metadata.

`jobs.json`, generated datasets, output metrics, logs, checkpoints, model weights, and observer credentials are excluded because they are execution state rather than source code.

