# Full exact-256 runs for the three smaller models

The user requested full training for DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-1.7B and the base Qwen2.5-3B after their timing profiles. Each run has 1,000 total optimizer updates: 256 history-building bootstrap updates followed by 744 updates at exact policy-version age 256. Each starts from its pinned initial model because the four-update profiles did not reach a 100-update recovery checkpoint. Profile optimizer steps and outputs are not reused as training state.

| Model | Pinned revision | Target node | Training / inference GPUs |
| --- | --- | --- | --- |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 | deep-chungus-5 | 4 / 5, all nine A100 40GB GPUs |
| Qwen/Qwen3-1.7B | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | deep-chungus-4 | 4 / 5, all nine A100 40GB GPUs |
| Qwen/Qwen2.5-3B | 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b | deep-chungus-7 | 4 / 3, seven working A100 80GB GPUs; eight reserved exclusively |

The 1.5B full run uses one additional inference worker compared with its 4+4 profile. This uses the full nine-GPU node while retaining four trainer ranks, 512 responses per update and the same learning protocol. Sampling execution and generation throughput can differ; it is not a bitwise replay of the profile. The node 7 seven-GPU fallback was explicitly authorized after a device failed CUDA initialization.

All scientific configuration is inherited from each verified profile: 64 prompts × 8 responses, 2,048 prompt and 8,192 response token limits, seed 42, learning rate 1e-6, 30 warmup updates, centered group advantages, GRPO clip 0.2, global token mean, no reference KL and zero weight decay. No zero-advantage filtering, shortened responses, skipped updates or intermediate evaluation is introduced. Model-specific native tokenization remains as validated in each profile.

`scripts/prepare_full_run.py` verifies profile model pins, frozen package hashes, source/runtime/data identity and question membership provenance. It changes only the output path and the selected inference worker count. It records the new configuration fingerprint and launch-file hashes in `full-run.json`. The frozen profile releases and prepared assets remain live dependencies of the full jobs and must not be removed.

`scripts/launch_full_run.py` verifies these bindings again inside the allocation, rejects occupied GPU memory, validates unique GPU identities and the exact prepared topology, and runs the collective/BF16/Flash Attention/vLLM kernel checks. It then executes the existing study CLI directly. `run_bounded_profile.py` is not part of these launches, so no four-update or five-hour profile cutoff applies. The imported official PrimeRL library and running 14B source remain unchanged.

Use account `grad-students`, background partition/QoS, full-node exclusive allocations and 45-day wall limits. Live account associations permit background and high-priority but not normal QoS; the existing 14B job occupies 8 of the 12 allowed high-priority GPUs, so another full node cannot start in that class. Background jobs can be preempted. Automatic Slurm requeue is disabled because safely restoring optimizer and rollout state requires an explicit complete checkpoint, a fresh output directory and matching identity. A new attempt must not overwrite an existing output or silently reset the policy-version clock. Checkpoints remain every 100 updates on NFS, with all 100-step milestones retained; metrics also go to XFS and Runboard. A preemption before the first checkpoint cannot be resumed from an intermediate state.

The 1.5B job is submitted with `afterany:2145014` so its current bounded timing profile can finish before full training starts on node 5. The other two nodes were idle at preparation time. Scheduler admission, submitted IDs and live startup evidence are recorded in the launch receipt after submission.

The base Qwen2.5-3B profile completed all four updates but every consumed response received zero reward and every advantage was zero. The user explicitly requested the full run after this was reported. Keep the requested base model and reward protocol; a running process is not proof of learning. Monitor reward diversity and noncontributing-token fraction. Timing from these short, zero-advantage responses cannot establish a useful full-training ETA.

## Submitted full runs

All three jobs were accepted at **2026-09-27 02:20:50 UTC**, using launch code from commit `34bed0f`. Effective scheduler fields were verified: background partition/QoS, account `grad-students`, exclusive single-node allocation, all node memory, 45-day limit and no automatic requeue.

| Model | Full-run job | Initial scheduler state | Output below the cluster root |
| --- | --- | --- | --- |
| DeepSeek 1.5B | 2145032 | Pending `afterany:2145014`; starts after its profile | `outputs/deepseek15b-exact256-seed42-v1` |
| Qwen3-1.7B | 2145033 | Running on deep-chungus-4 | `outputs/qwen3-1p7b-exact256-seed42-v1` |
| Qwen2.5-3B | 2145034 | Running on deep-chungus-7 | `outputs/qwen25-3b-exact256-seed42-v1` |

Control folders under `launches/` are `deepseek15b-full-20260927`, `qwen3-1p7b-full-20260927` and `qwen25-3b-full-20260927`. Each records `study.json`, `full-run.json`, `submission.json` and per-job health/launch receipts. The [combined submission receipt](../diagnostics/full-runs-20260927.json) contains exact commands, config changes, identities and file hashes. Twenty-two local full-run, profiling and recovery tests passed, along with Ruff and shell syntax.

The existing 30-minute monitor now follows these full jobs and the 14B production/continuation jobs. It must not treat update 4 as a stopping condition for any full run. The old 1.5B profile remains bounded at four updates and retains its original source and outputs.

## Startup checks at 02:26 UTC

Qwen3-1.7B job 2145033 passed all nine free-memory/BF16 probes and the collective sum 45, Flash Attention backward and vLLM RMSNorm checks. Its resolved configuration has 1,000 updates and lag 256; its inference servers were serving generation requests. Runboard ID: `b0c2d25f0bfe4ff2bbc37219a4537b98`.

Qwen2.5-3B job 2145034 passed the corresponding seven-GPU checks with collective sum 28. Device 7 again could not initialize and was excluded. Its resolved configuration also has 1,000 updates and lag 256; worker initialization was still in progress at this observation. Runboard ID: `4dbd88e541aa46639d6fd087b9d59b78`.

DeepSeek 1.5B job 2145032 was pending its profile dependency, so its full nine-GPU runtime validation remains pending. No completed optimizer update is claimed for the new full runs at this startup observation. See the [startup receipt](../diagnostics/full-runs-startup-20260927.json) for configuration, source identities, scheduler state and hardware results.
