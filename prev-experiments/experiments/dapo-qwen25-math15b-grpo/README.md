# Qwen2.5-Math-1.5B DAPO GRPO

This is an independent run of the same training and evaluation setup as the existing Qwen2.5-Math-7B experiment. The model is the base `Qwen/Qwen2.5-Math-1.5B`, pinned to revision `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`. The 7B run is unchanged.

The configuration keeps 1,000 updates, seed 42, batch size 64, groups of eight completions, four FSDP2 trainer GPUs, two vLLM data-parallel replicas, learning rate `1e-6`, PPO clipping at 0.2, and the same evaluation benchmarks and intervals. The total sequence budget is 4,096 tokens with at most 3,072 generated tokens. Reward and custom loss code are identical to the 7B experiment.

Preparation downloads the pinned model into `/mnt/nfs/home/mohamadzbib/projects/models/Qwen2.5-Math-1.5B-4a83ca6e`. It rebuilds the tokenizer overlay and data manifest, measures prompt lengths using this model's tokenizer, and verifies that every retained prompt, answer, source ID, row order, and exclusion record agrees with the prepared 7B data. A comparison mismatch stops preparation. No 7B manifest hashes are reused as hashes of the new model or scripts.

Production writes to `/mnt/nfs/home/mohamadzbib/projects/rl-infra/outputs/dapo-qwen25-math15b-grpo/qwen25-math15b-grpo-seed42`, exposed through `/mnt/xfs/home/mohamadzbib/projects/models/qwen25-math15b-grpo-seed42`. Observer and recovery state have separate directories with the same 1.5B experiment prefix. Completed checkpoints retain trainer and orchestrator state and are saved every 25 steps with the same retention policy as the 7B run.

The Opik project is `qwen25-math15b-grpo` in workspace `mohamad-zbib-0046`. Its [Comet training dashboard](https://www.comet.com/opik/mohamad-zbib-0046/projects/01a0a4f7-122a-76ce-b024-943a82b62e73/dashboards?dashboardId=01a0a4f7-1347-74b0-8bfc-f9a3963cb5bd&dashboard_time_range=past24hours) has 12 widgets copied from the 7B training monitor. The observer reads the existing private credential store and overrides only its own process's project name. It does not edit the credential store or send 1.5B metrics to the 7B project. Prime's local `metrics.jsonl` remains the durable metric source. Charts populate when the production observer begins uploading metrics; creating the dashboard alone does not establish that training is running.

The submitted chain runs independently on `deep-chungus-9` under the `grad-students` account and `background` partition:

| Job | Stage | Resources | Dependency and current recorded state |
| --- | --- | --- | --- |
| `2140587` | Model download, data preparation, CPU validation, and Prime dry-run | 8 CPUs, 32 GB memory | Completed, exit `0:0` |
| `2140588` | Six-GPU health check and two-update training smoke test | 6 A100s, 64 CPUs, 512 GB memory | Completed, exit `0:0`; two optimizer updates and paired checkpoint verified |
| `2140589` | Full 1,000-update production run | 6 A100s, 64 CPUs, 512 GB memory | Running; started automatically after smoke passed |

Both GPU stages keep four trainer GPUs and two vLLM replicas on the same node. All three submissions use `--time=0`, `--requeue`, and `exec bash`, with no scheduled time-limit warning. Production releases its allocation when it completes or exits on an operational failure; it has no artificial wall-clock cutoff. The dependency chain blocks production if preparation or smoke validation fails. Preparation and GPU smoke validation completed successfully. Production is starting, and its authenticated Comet startup trace was verified. Production training metrics are pending.

`jobs.json` records the exact submitted commands, dependencies, and dashboard identifiers. Slurm logs are under `/mnt/nfs/home/mohamadzbib/projects/rl-infra/logs/dapo-qwen25-math15b-grpo`, with separate `prepare-2140587`, `smoke-2140588`, and `production-2140589` stdout/stderr files. The existing 7B experiment continues under its own allocation, outputs, checkpoints, and Comet project.

CPU preparation 2140587 and the six-GPU end-to-end smoke 2140588 completed with exit 0:0. The smoke produced two finite-gradient optimizer updates and a complete trainer/orchestrator checkpoint at step 2. Production 2140589 started automatically on deep-chungus-9; startup is being checked. Smoke checkpoints are also linked from `/mnt/xfs/home/mohamadzbib/projects/models/qwen25-math15b-grpo-smoke-2140588`.
