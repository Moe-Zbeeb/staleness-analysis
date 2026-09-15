# Qwen3-14B DAPO GRPO

Independent run of `Qwen/Qwen3-14B`, revision `40c069824f4251a91eefaf281ebe4c544efd3e18`. This checkpoint is post-trained and supports thinking; it is not a base model. Non-thinking mode preserves the existing 3,072 generated-token budget and greedy benchmark protocol. The reference prompt template and problem instructions are retained, with Qwen3's empty `<think>...</think>` prefix appended at generation time.

Same experiment: 17,005 cleaned DAPO training rows, seed 42, 1,000 updates, batch 64, group 8, AdamW learning rate 1e-6, custom PPO clipping loss at 0.2, no KL penalty, 4 FSDP2 trainer GPUs and a vLLM replica split over 2 GPUs. Dataset membership and row order are checked against the original run, with fresh Qwen3 prompt-token counts. The model-family and prompt-prefix differences must be considered when comparing results.

The current execution profile targets six 40GB A100s on deep-chungus-4, with 64 CPUs and 384GB host RAM. To fit this hardware, optimizer states are offloaded to CPU memory, FSDP reshards after the forward pass, and inference uses tensor parallelism 2 with data parallelism 1. The GRPO objective, optimizer hyperparameters, data, seed, global batch size and update target are unchanged; numerical execution and throughput can differ. The original 80GB profile is archived remotely under execution-history/80gb-before-chungus4. A fresh H3 probe (2140602), using allocated GPUs 0–5, failed CUDA initialization. Preparation and GPU smoke validation gate production. Submissions use unlimited Slurm wall time, requeue and checkpoint resumption; operational failures and preemption remain possible.

Base model: `/mnt/xfs/home/mohamadzbib/projects/models/Qwen3-14B-40c06982`. Production checkpoints: `/mnt/xfs/home/mohamadzbib/projects/models/qwen3-14b-grpo-seed42`. Comet project: `qwen3-14b-grpo`. Exact job IDs and validation states are recorded in jobs.json.

Download `2140595` completed with exit `0:0`; CPU preparation `2140597` completed with exit `0:0` in 2m55s. Model/data/tokenizer hashes, matching dataset membership/order, the non-thinking prefix, loss/reward checks and Prime dry-run all passed. Maximum training prompt length is 1,023 tokens, within the existing 1,024-token limit.

Chungus-4 configuration preparation is job `2140605`. Existing GPU smoke `2140598` now targets Chungus-4 and depends on that preparation. Production `2140599` targets the same node and depends on successful completion of the smoke test. Preparation completed with exit 0:0, including the Prime dry-run and unchanged data hashes. Smoke job 2140598 is running on six allocated A100-40GB GPUs; production remains gated on smoke success.

[Comet dashboard](https://www.comet.com/opik/mohamad-zbib-0046/projects/01a0a574-ba0f-7142-9715-44e23f69369a/dashboards?dashboardId=01a0a574-df9d-73e0-9ac8-a2e99bc7966d&dashboard_time_range=past24hours) is created and verified with 12 panels; it will receive metrics when production starts.
