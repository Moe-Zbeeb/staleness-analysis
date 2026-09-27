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
