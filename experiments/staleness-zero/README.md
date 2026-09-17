# Zero-staleness GRPO experiment

The current study direction has moved to the [cap-4/6/8 sweep](../../docs/staleness-cap-sweep.md). These prepared cap-0 files are retained as references and are not part of that sweep.

Prepared on September 16, 2026. The four production configurations are frozen and locally checked. No new cluster job has been submitted. GPU smoke validation is still pending.

## Treatment

At training update `t`, accept only rollouts whose starting and ending policy versions equal `t - 1`. Set `orchestrator.max_off_policy_steps = 0`, compared with the existing cap of `2`. Start each model from the same original pinned model snapshot, with fresh optimizer and run state.

PrimeRL `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1` checks queued rollout freshness before shipping a batch. Tightening this existing setting is sufficient; no loss or framework patch is required. The pipeline may generate and discard outdated work while waiting for an updated inference policy. Those discarded rollouts are not training data. Matching policy versions also does not guarantee bitwise-identical log probabilities between vLLM and the trainer.

## Frozen runs

| Model | New experiment directory | Trainer + inference GPUs |
| --- | --- | --- |
| Qwen2.5-Math-1.5B | `../dapo-qwen25-math15b-grpo-stale0` | 3 + 2 |
| Qwen2.5-3B | `../dapo-qwen25-3b-grpo-stale0` | 4 + 3 |
| Qwen2.5-Math-7B | `../dapo-qwen25-math7b-grpo-stale0` | 8 + 1 |
| Qwen3-14B | `../dapo-qwen3-14b-grpo-stale0` | 7 + 2 |

Each new run name ends in `-seed42-stale0`. Outputs, model links, observer state, recovery state, and Opik project names are separate from the baseline. The 14B retains its corrected non-thinking tokenizer, CPU optimizer offload, and tensor-parallel inference.

Production settings retain 1,000 updates, batch size 64, group size 8, learning rate 1e-6, clip epsilon 0.2, the same exact-answer reward and GRPO loss, and no reference KL penalty. Training and standard evaluation allow 3,072 generated tokens with a 4,096-token context. Minerva and OlympiadBench retain their 2,048-token evaluation cap. Dataset rows, ordering, tokenizer bytes, model revisions, evaluation sources, sampling settings, and evaluation schedule are preserved per model.

Seed settings are preserved. An asynchronous staleness intervention can change accepted sample order and floating-point results; identical seed settings do not make paired trajectories identical.

## Baseline provenance and comparison limits

The source and resolved runtime configurations were fetched from the cluster. Every directory contains `baseline/`, `comparison.json`, and `source-manifest.json`. The static guard reconstructs the expected production configuration from its frozen reference and accepts only the cap change plus run identity/path changes. The original experiment directories were not edited.

The 7B reference uses its verified final execution profile, whose config SHA-256 is `d30606c47ad32c33b748f4814d44774f343c1bc92ddd2df06c4270005721fde4`. Its historical cap-2 run used 4 trainer + 2 inference GPUs before resuming at update 975 with 8 + 1. A new 8 + 1 run is therefore not a perfectly controlled comparison with that entire historical run. A strict causal comparison needs a fresh cap-2 control on the same fixed topology. That additional training has not been prepared or launched as part of this treatment.

The 1.5B and 3B final baselines used partial allocations of 5 and 7 GPUs. The saved September 16 inventory had no five-GPU A100 node; its seven-GPU node was down. Their matching full-node placement is therefore unresolved. The submission helper refuses to change these topologies or allocate only part of a larger node automatically. At launch, either establish an authorized matching partial allocation with current capacity, or design fresh matched cap-2/cap-0 controls on an available full node. Do not silently change GPU count and call staleness the only changed setting.

## Validation and launch sequence

1. Deploy the four new directories and this `staleness-zero` directory under `/mnt/nfs/home/mohamadzbib/projects/rl-infra/experiments/` without overwriting existing runs.
2. Run local static checks and the tests below. They do not require model weights or GPUs.
3. On an authorized GPU allocation, `scripts/smoke_job.sh` calls `prepare_job.sh`. It copies and hash-verifies the already prepared baseline data and tokenizer, creates a new manifest, runs the existing complete experiment validator, and resolves the PrimeRL production config with `--dry-run`. It does not regenerate datasets or tokenizer templates.
4. The smoke run performs five updates with the production batch/group sizes, concurrency settings, and 3,072-token training limit. It shortens the evaluation to four MATH500 examples and warmup to one update; these overrides affect only the smoke test.
5. The smoke guard requires finite gradient/loss metrics, all six shipped-cohort staleness metrics equal to zero at every update, every effective trace's raw `policy.start` and `policy.end` equal to `step - 1`, and both final checkpoint components. Checking only the `train/all` traces would be incorrect because that file also includes discarded work.
6. `train_job.sh` refuses to start without matching smoke evidence. It initializes from the original model and can resume only this new experiment's own checkpoints. A final audit checks all 1,000 updates before writing the training-finished marker.

The existing full-node health probe exercises BF16 backward and NCCL with a world-size-derived expected result. Visible GPU ordering is derived from the allocated IDs, including non-contiguous IDs.

Local tests, from this directory, using Python 3.11 or newer:

```bash
python scripts/test_staleness_guard.py -v
python scripts/staleness_guard.py static --experiment-root ../dapo-qwen3-14b-grpo-stale0
```

For a future authorized cluster submission, activate the pinned environment and use the submission helper. Without `--submit`, it only reports the live candidate nodes, QoS capacity, queue, and effective command. The helper requires high-priority account/partition/QoS, exclusive allocation, and the exact full-node GPU count; it verifies and records the submitted Slurm fields. Five- and seven-GPU placement may fail for the reasons above. The QoS cap can also serialize the nine-GPU runs.

```bash
source /mnt/nfs/home/mohamadzbib/projects/rl-infra/env.sh prime-rl
uv run --no-sync python /mnt/nfs/home/mohamadzbib/projects/rl-infra/experiments/staleness-zero/scripts/submit.py --experiment-root /mnt/nfs/home/mohamadzbib/projects/rl-infra/experiments/dapo-qwen3-14b-grpo-stale0 --stage smoke
```

Add `--submit` only when actually launching. After the smoke job passes, use `--stage train --submit` for production. Inspect the startup health output; Slurm RUNNING status alone is not a passed GPU test.

## Analysis

Compare reward and held-out accuracy at matched update counts and token budgets. Report training truncation, accepted rollout lengths, generated versus accepted tokens, discarded work, policy-wait time, throughput, and wall time separately. Zero staleness can reduce useful asynchronous overlap. A longer evaluation token budget should remain a separately reported diagnostic, applied equally to both arms.

One seed per model is a pilot comparison; it does not establish uncertainty across seeds. The historical topology changes must be disclosed when interpreting an effect as caused by staleness.
