# Qwen2.5-3B: short-IPC node-local launch

**Job 2145180 was cancelled during GPU startup at the user's request to replace Qwen2.5-3B with DeepSeek-R1-Distill-Qwen-1.5B. It completed no updates.**

Job **2145180** replaced failed startup **2145176**, which generated no archived rollouts and committed no updates. The new job uses high-priority partition/QoS, account `grad-students`, the complete configured eight-A100-80GB node `deep-chungus-7`, and the authorized seven-healthy-GPU topology: four trainers and three inference workers. No automatic requeue is enabled.

The correction changes only temporary socket paths. `TMPDIR` and `VLLM_RPC_BASE_PATH` now use a private short `/tmp/st-<uid>-<workspace-hash>` directory. Linux Unix-domain addresses must fit 107 bytes; the previous deeply nested temporary directory did not. A real ZeroMQ IPC bind now passes during preflight, before GPU startup. Model files, dataset, tokenizer, grader, GRPO settings, exact k=256 schedule, 1,000 updates and checkpoint cadence of 100 remain unchanged. Thirty-five focused tests and Ruff passed. Source commit: `28d1afe`.

The prepared workspace and assets remain at `/tmp/staleness-storage-v2/mohamadzbib/qwen25-3b-exact256-80gb-seed42-v4-local`. The new active output is its `runs/qwen25-3b-exact256-80gb-seed42-v5-local` child. The failed v4 output and launch evidence are preserved. Shared backups use the new `outputs/qwen25-3b-exact256-80gb-seed42-v5-local` NFS directory and matching XFS metric directory. Launch controls are under `launches/qwen25-3b-full-hp-local-ipc-20260927`, with `-logs` appended for Slurm logs.

This is **Qwen/Qwen2.5-3B base**, not a native thinking checkpoint. The actual prompt asks for step-by-step reasoning and a final `\boxed{}` answer. The native chat template ends at the assistant prefix without inserting `<think>`. The grader has `reasoning_required=false`, so missing `</think>` alone is not penalized. This does not establish response quality or a nonzero learning signal; inspect generated responses and rewards after startup.

At submission the job was RUNNING, but application startup, first-update throughput and the first full recovery checkpoint were not yet verified. See the [submission evidence](../../diagnostics/node-local-v5-20260927.json) and [storage guide](../node-local-storage.md).
