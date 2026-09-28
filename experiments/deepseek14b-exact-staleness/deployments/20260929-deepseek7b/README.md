# DeepSeek-R1-Distill-Qwen-7B preparation

The user requested full k256 and k0 studies with the same DAPO17k dataset, 32 responses per update (four questions times eight responses), 1,000 updates, 6,144 response tokens, 2,048 prompt tokens, seed 42 and unchanged GRPO/optimizer settings. Both training jobs must use high-priority partition and QoS. The k256 learner must run on A100 80GB GPUs with a separate bootstrap historical-inference pool.

The verified per-user high-priority limit is 12 A100 GPUs total, presently occupied by the running 1.5B k256 learner and history worker. The question of sequential full-quota studies versus concurrent smaller layouts is pending user clarification. No 7B training GPU jobs have been submitted, and no existing job has been stopped or reprioritized. The preparation configuration is an asset-validation configuration, not a finalized allocation.

CPU-only preparation job **2145682** was submitted with four CPUs, 24 GiB RAM, two-hour limit, high-priority partition/QoS and no requeue. It started on deep-chungus-1. It downloads the pinned model, records file hashes, validates native tokenizer parity, validates every dataset reference, requires exactly the same included question IDs and order as the current 1.5B study, and writes a completion receipt only after preflight passes. At this record's creation it was downloading; GPU memory and full-update execution for 7B are not yet verified.

Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, pinned revision `916b56a44061fd5cd7d6a8fb632557ed4f724f60`.

Dataset: `zbeeb/Staleness-GRPO-DAPO-Math-17k`, same 17,005-row locked source bytes as the active 1.5B runs.

The prepared release starts from the active 1.5B k0 release, so both k0 and k256 can use independent local inference replicas and filesystem weight transfer. The three application files implementing that existing k0 pool option are preserved under `release/src` here, along with the 7B model constants. The current backup race fix is included. Official imported PrimeRL and vLLM source remain unchanged; no GRPO loss or reward-policy edits were made for this preparation.

Cluster controls: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek7b-dapo17k-b32-hp-20260929-v1`.
