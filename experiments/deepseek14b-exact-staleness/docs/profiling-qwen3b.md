# Qwen2.5-3B timing profile on node 7

The user requested the exact base model [`Qwen/Qwen2.5-3B`](https://huggingface.co/Qwen/Qwen2.5-3B), pinned here to revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`. This does not substitute the Instruct checkpoint. The model retains its native tokenizer, supplied chat template, EOS behavior and tied embeddings. Its template includes Qwen's default system message and assistant prefix; no DeepSeek thinking prefix is added.

This is the same four-update bootstrap timing protocol as the [1.5B profile](profiling-small-model.md). The frozen 14B run provides the configuration and adapter source. Keep 64 prompts × 8 responses, 2,048/8,192 prompt/response caps, seed 42, the optimizer and GRPO settings, logging, exact lag 256, and the configured 1,000-update schedule. An external supervisor ends the profile after four committed updates and their learner timing records, or five hours after launch. The job allocation is bounded at six hours. No intermediate evaluation or checkpoint is due during this prefix.

## Requested hardware and authorized fallback

Target `deep-chungus-7`, account `grad-students`, partition/QoS `background`, exclusive allocation, eight requested A100 GPUs, 128 CPU slots, all node memory and no requeue. The node previously failed eight-device CUDA initialization despite Slurm and NVML listing eight cards. The user explicitly authorized seven healthy GPUs if required: four trainers and three inference workers. The normal layout remains four plus four when all eight devices pass.

`scripts/probe_allocated_gpus.py` checks every allocated device in its own bounded process with BF16 forward/backward. `scripts/profile_qwen3b_job.sh` rejects fewer than seven healthy devices or duplicate GPU identities. It then runs the existing full collective, BF16, Flash Attention and vLLM health probe on the selected seven or eight devices before preparing any model or starting training. A seven-rank all-reduce must sum to 28; eight ranks must sum to 36. Device selection comes from the actual allocation, and the allocation receipt records every selected and unused device. No driver change, reset, node administration or access outside the allocation is performed.

Using three inference workers changes generation throughput and sampling execution. It does not change the four-rank learner, cohort size, loss or policy-version schedule. Compare learner timing, generation timing and token counts separately. Hardware, model family and native prompt differences prevent interpreting this as a controlled model-size-only comparison.

## Adapter changes

Official PrimeRL remains pinned and unmodified. The generated isolated adapter changes the two model identity constants and the model-specific checks inside `dataset/assets.py`. For Qwen, those checks compare prepared token IDs, decoding, rendered prompts and BOS/EOS against the pinned original Qwen tokenizer. The original DeepSeek-specific thinking suffix and BOS/EOS constants are not appropriate for this base model. The dataset, grader, optimizer, GRPO loss, queue and trainer integration files remain byte-identical to the frozen 14B source.

Preparation downloads and hashes the pinned model, verifies native tokenizer parity, creates a new data manifest, and requires the retained question IDs and order to match the 14B baseline. It allows only isolated asset/output paths and the authorized inference GPU count to differ in the study configuration. All source and configuration differences are recorded in `preparation.json`; the running 14B and 1.5B releases are untouched.

The job writes normal metrics to Runboard and XFS, with original logs and rollouts on NFS. Its name is `qwen25-3b-exact256-timing-<job-id>`. `work/timing-summary.json` distinguishes completion of the bounded profile from an early exit or timeout; it is also copied to the run's XFS mirror. Runboard can label the deliberately interrupted longer training process as killed when the four-update profile finishes. No exact-256 steady-phase or checkpoint timing is claimed from four bootstrap updates.

Validation includes source-difference and tamper guards for both model profiles, successful bounded termination, early failure and timeout. Actual CUDA and model startup results must be recorded separately from these local checks.

## Submitted job and startup evidence

Job **2145015** started on `deep-chungus-7` under background partition/QoS with source commit `a0c9e27`. Devices 0–6 passed BF16 forward/backward; device 7 failed CUDA initialization and is unused. The selected seven A100 80GB PCIe GPUs also passed the collective sum of 28, BF16 backward, Flash Attention backward and vLLM RMSNorm probes. Inference uses devices 0–2 and training uses devices 3–6.

Native Qwen tokenizer parity passed all five probes. Data preparation retained the same 37,703 questions in the same order as the 14B baseline. At **2026-09-27 00:37:36 UTC**, the four-rank learner and three-worker inference were starting, the controller had initialized, and Runboard had registered the run. No optimizer update had completed at that observation; timings and successful full-cohort execution remained unverified. This is a timestamped launch record, not a live status report.

The [submission and startup receipt](../diagnostics/profiling-20260927-qwen3b.json) records allocation, device checks, pinned model revision, configuration differences and run identity. The 14B production job and 1.5B profiling job were not modified. PrimeRL itself remains unchanged.

Cluster paths:

- Control directory: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/profiling/20260927-qwen3b-node7`.
- Isolated release and resolved configuration: `work/release` and `work/study.json` below that directory.
- Run output: `work/qwen25-3b-exact256-timing-2145015`.
- Bounded timing result: `work/timing-summary.json`, written when the profile exits.
- XFS mirror: `/mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics/qwen25-3b-exact256-timing-2145015`.
- [Runboard](https://runboard-cloudflare.mbz02.workers.dev): project `staleness-analysis`, name `qwen25-3b-exact256-timing-2145015`, run ID `f52d563535754063b5afcbe07f26e30a`.

## Failure after the first generated cohort

Job **2145015** failed after 18 minutes 11 seconds, before completing any optimizer update. It generated one 512-response bootstrap cohort containing 360,188 output tokens in 98.59 seconds. During the first backward pass, trainer rank 2 could not allocate another 136 MiB: only 121.19 MiB was free. The trainer held about 8.01 GiB while another process held 71.11 GiB. This was not the intended four-update termination and was not reported by Slurm as preemption.

A subsequent short allocated diagnostic found PID **2423437**, `VLLM::EngineCore`, still occupying 72,816 MiB on GPU UUID `GPU-99e6d841-8f03-96c4-7573-c45c3e4026cc`. It belongs to `mohamadzbib`, started on September 24 at 05:35:30 in the cluster's local time, has parent PID 1, and uses the older `/mnt/nfs/home/mohamadzbib/projects/rl-infra/repos/prime-rl` directory and environment. No Slurm job environment was present. It is separate from this profile's recorded trainer and inference worker PIDs. GPU UUIDs distinguish CUDA-visible numbering from NVML indices. The [failure and ownership receipt](../diagnostics/profiling-20260927-qwen3b-failure.json) preserves the evidence.

The earlier tiny BF16 health check passed despite insufficient memory for training. New profiling preflight requires at least 90% free CUDA memory before its backward probe, records free/total bytes, and stops on an occupied GPU instead of silently treating it as a faulty device eligible for the seven-GPU fallback. New Qwen wrappers also set `CUDA_DEVICE_ORDER=PCI_BUS_ID` before both health checks and model launch, matching the PrimeRL worker setting. The loss, optimizer, cohort size and staleness schedule are unchanged. Running jobs and frozen releases are unchanged.

A fresh retry directory is prepared at `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/profiling/20260927-qwen3b-node7-v2`. After the user authorized cleanup and relaunch, diagnostic job 2145025 verified the old server's UID, start time, parent and working directory. SIGTERM did not release it; SIGKILL stopped that same verified process. All eight NVML devices then reported 1 MiB used and no compute processes. Retry job **2145026** was submitted at 2026-09-27 01:20:13 UTC with the memory guard from commit `b787876`, background partition/QoS, exclusive node 7, eight requested GPUs and a six-hour limit. The original failed run and its outputs remain intact. Startup validation is recorded separately below.

The occupied-device guard was verified on the cluster in diagnostic job **2145023**: it rejected CUDA device 5 with 8,295,481,344 of 85,097,971,712 bytes free before running backward. An earlier eight-process diagnostic with eight CPU slots, 8 GiB host memory and a 60-second child timeout timed out on all probes; it did not validate readiness. A clean full-node health check remains required for the retry. Six local profile tests, Ruff, wrapper shell syntax and embedded probe syntax passed.
