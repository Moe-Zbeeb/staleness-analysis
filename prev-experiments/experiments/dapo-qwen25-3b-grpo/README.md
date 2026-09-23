# Qwen2.5-3B DAPO GRPO

Independent general-base Qwen/Qwen2.5-3B run, revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`. This is not a Qwen2.5-Math or Instruct model. Comparisons with the 1.5B and 7B runs therefore include model-family differences.

Uses the same DAPO rows, seed 42, prompt template, deterministic reward, custom PPO clipping loss, 1,000 updates, learning rate 1e-6, batch size 64, group size 8, 4 trainer GPUs and 2 vLLM replicas. Data preparation verifies prompt/answer identities and row order against the original 7B run while recomputing token lengths with the 3B tokenizer.

CPU preparation and a two-update, six-GPU smoke test gate production. Planned node: deep-chungus-5, 64 CPUs and 256 GB RAM. Slurm time=0 means no wall-clock cutoff; operational failure and preemption remain possible. The existing wrapper resumes complete paired checkpoints on restart. Production checkpoints are saved every 25 updates and exposed under projects/models/qwen25-3b-grpo-seed42. Model weights are under projects/models/Qwen2.5-3B-3aab1f19.

Comet project: qwen25-3b-grpo. Submission and validation records will be saved in jobs.json.

Preparation `2140590` completed with exit `0:0`. Smoke `2140591` completed in 13m32s with exit `0:0`: both finite-gradient optimizer updates, before/after evaluation, six-GPU BF16/NCCL checks and the paired step-2 checkpoint passed. Peak trainer memory was 19.8 GiB. Production `2140592` is running its startup on deep-chungus-5 with six A100-PCIE-40GB GPUs.

[Comet training dashboard](https://www.comet.com/opik/mohamad-zbib-0046/projects/01a0a553-5b41-756c-9e02-9d7a5e0cf69d/dashboards?dashboardId=01a0a553-8621-707a-8291-bce3881f9aea&dashboard_time_range=past24hours) has 12 verified panels. Metrics populate once the production observer starts. Smoke checkpoints are exposed at `/mnt/xfs/home/mohamadzbib/projects/models/qwen25-3b-grpo-smoke-2140591`.

Production startup authenticated to its own Comet project successfully; startup trace `01a0a564-ebcf-7a27-a59a-738415fc29ae` was verified through the API. The observer is ready. Production optimizer metrics are pending while model workers initialize.
