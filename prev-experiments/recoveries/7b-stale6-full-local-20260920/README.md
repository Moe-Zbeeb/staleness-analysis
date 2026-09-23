# 7B staleness ≤6: full chungus-9, local metrics

The user requested resuming this run on the full deep-chungus-9 node with Comet disabled. The existing normal-priority authorization remains in effect.

Job **2142129** started at **12:04:14 UTC on September 20, 2026**, with all **eight A100 80GB GPUs**, an exclusive node request, 64 requested CPUs (128 allocated by exclusivity), and 384 GiB requested host memory. It resumes the original four-trainer checkpoint **125**, restoring the original four inference replicas. Training ends at step 1000.

The launcher never loads Comet credentials, starts an Opik uploader, or waits for its readiness. The unchanged file monitor writes persistent metrics to the original run's `metrics.jsonl`; checkpoint, rollout, evaluation and component logs retain their existing paths. Existing historical Comet records remain untouched. Completion markers use the existing observer directory solely as local run metadata and require the original final-checkpoint/evaluation audit.

## Previous failure and bounded change

Job 2142075 failed while waiting 1,200 seconds for the trainer's startup weight publication at version 125. Its inference server had become healthy, but the initial trainer broadcast did not appear before the deadline. This identifies the immediate timeout, not the underlying cause of delayed trainer startup.

The only resolved configuration difference from the frozen original eight-GPU experiment is `orchestrator.ckpt.wait_for_weights_timeout = 7200`. The shared Prime-RL repository, training data, model identity, optimizer, scheduler, staleness bound, generation settings and evaluation schedule are unchanged. No cap-4 runtime patch is applied. All eight original Slurm CUDA selectors are retained; PRIME reorders the same set into inference GPUs 4–7 and trainer GPUs 0–3.

## Validation

Five CPU regression tests passed, covering full-node CUDA identity preservation despite different Linux minor numbering, partial allocations, duplicate UUIDs, mismatched PCI identities, and pre-existing compute processes. Python compilation, shell syntax and all 15 deployed package hashes passed.

At 12:05 UTC, allocated configuration/checkpoint validation passed: trainer step 125, orchestrator next step 126, and 1,381 model/optimizer/scheduler metadata entries. Available host RAM was approximately 942 GiB. At **12:08:47 UTC**, all eight GPU UUID/occupancy checks and the eight-rank BF16/NCCL health probe had passed, with collective sum **36**. The launcher captured checkpoint 125 and launched PRIME with Comet explicitly disabled. Complete trainer weight restoration and the first new optimizer step were not yet verified. See [startup evidence](startup-2142129.json) and [GPU health](gpu-health-2142129.json).

## Evidence

- [Submission and effective Slurm fields](training-submission.json)
- [Immutable package manifest](package-manifest.json)
- [Continuation specification](spec.json)

Package manifest SHA256: `95e65d3857aaf07e8929db8bdeac67e606ab36965214a35837f0a3dc017a3a3e`.

Remote package: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/recoveries/7b-stale6-full-local-20260920`.

Remote run: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/outputs/dapo-qwen25-math7b-grpo-stale6-8h100/qwen25-math7b-grpo-seed42-stale6-8h100`.
