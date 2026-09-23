# Qwen2.5-3B staleness ≤6: startup failure diagnosis

## Result

Job `2141854` failed with exit `1:0` after 18 minutes 8 seconds on deep-chungus-9. The failure occurred in the smoke run before any training updates. The output directory contains only the smoke run, an empty metrics.jsonl, and no step checkpoints. No production run, checkpoint, or evaluation result exists for this attempt. The latest queue has no active ≤6 replacement.

## Confirmed failure mechanism

The four-GPU configuration assigned two GPUs to training and two to inference. Inference resolved to tensor parallelism 1, data parallelism 2, and two API server processes. Both inference engine processes failed while waiting for the API frontend's initialization response:

> Did not receive response from front-end process within 5 minutes

Installed vLLM source sets `HANDSHAKE_TIMEOUT_MINS = 5` in `vllm/v1/engine/core.py:93`; `startup_handshake` uses this constant at line 1234. This is separate from the orchestrator's 1,800-second readiness timeout and the router's 4,200-second health timeout. Increasing those outer timeouts cannot fix this specific failure.

## Evidence and timeline

Times below are copied from the same component-log clock; they are not relabeled as UTC.

| Time | Event |
|---|---|
| Before 18:01:26 | Four A100 80GB GPUs passed BF16 backward and NCCL all-reduce; world size 4, sum 10. |
| 18:01:26 | Launcher started inference, trainer, orchestrator and environments. |
| 18:02:23 | Training and evaluation environments ready; orchestrator waited for inference. |
| 18:03:04 | Trainer finished loading weights and constructing optimizer; waited for weight broadcast. |
| 18:04:44 | vLLM launched two API processes and the data-parallel coordinator. |
| 18:06:11 | Both API processes were still constructing their model configuration. |
| 18:07:18 | API/engine logs recorded scheduler configuration. |
| 18:10:51 | Both inference engines hit the hardcoded five-minute handshake timeout. |
| 18:11:17 | Both API processes logged further configuration progress, 26 seconds after the engines had failed. |
| 18:11:24 | Launcher detected inference failure and terminated the other components. |

This establishes excessive frontend initialization latency relative to the engine handshake deadline. It does not establish the exact slow operation: there are no per-import timings or stack samples from the blocked frontend interval. Shared-filesystem reads and concurrent startup are plausible contributors, not proven root causes. DP2 is not intrinsically unsupported; the evidence concerns this startup attempt.

No CUDA out-of-memory error or port-bind failure was recorded. GPU health tests and trainer weight loading passed. Trainer SIGTERM was cleanup after inference failed, not an independent trainer crash. The staleness limit had not yet been exercised.

## Recovery configuration to validate

A practical next attempt is two trainer GPUs plus one inference instance sharded over two GPUs: TP2, DP1, one API server. Prime's server selects its single-API-server path when the resolved API count is one; this avoids the failing multi-frontend startup path. This changes inference topology, so preserve the failed experiment and validate a separately recorded attempt, including weight broadcast and the 25-step smoke gate before production. Keep cap6 and scientific training settings unchanged. A real smoke run is needed before calling this fix verified.

The separately queued ≤8 job `2141858` already uses this proposed topology, but it has not run and therefore does not yet validate it.

If retaining DP2 for a controlled diagnosis, profile frontend initialization and use an experiment-scoped, source-checked handshake-timeout change only if needed. Do not patch the shared vLLM installation used by active jobs.

## Current resources and actions

At the live check, deep-chungus-9 had 6/8 GPUs allocated, leaving two. A four-GPU recovery cannot start there immediately. Job 2141858 (3B ≤8) is pending Resources. This diagnosis submitted no job, changed no training settings, and interrupted no active run.

## Provenance

- Live check: 2026-09-17T23:52:24.568037+00:00
- `tmp/diagnose-3b-stale6-20260918/snapshot.json`: Slurm queue/accounting, node resources, resolved configurations and logs.
- `tmp/diagnose-3b-stale6-20260918/source.json`: installed handshake source and run-directory/checkpoint inventory.
- `tmp/move-3b-stale8-20260918/startup_source.json`: installed Prime single/multiple API dispatch and vLLM timeout source.
- Failed experiment: `staleness-analysis/experiments/dapo-qwen25-3b-grpo-stale6-4gpu/`.
