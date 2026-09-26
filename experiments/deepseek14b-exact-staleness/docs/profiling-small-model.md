# Bounded 1.5B timing profile

This job measures the first four complete updates of `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, pinned to Hugging Face revision `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`. The user chose 1.5B only; no 3B job is submitted.

The baseline is the frozen configuration and source used by 14B job `2144963`: `releases/correctness-v2-20260926`, source commit `5ad56dedbd4bfd3a4e3bb2f39ac4e56d0de274d8`. The profiling job requires one exclusive eight-A100-80GB node, with four inference GPUs and four trainer GPUs. Background partition/QoS is selected as requested; preemption can interrupt the measurement. Automatic requeue is disabled to prevent duplicate writers.

All training settings stay identical: 64 prompts × 8 responses, 2,048 prompt and 8,192 response token limits, 1e-6 learning rate with 30-update warmup, seed 42, zero weight decay, centered advantages, clipped GRPO, the same logging, native tokenizer preparation, and no online evaluation. Model, prepared assets, data manifest and output paths are isolated. The 1,000-update budget, exact lag 256 and checkpoint interval 100 remain configured. An external supervisor stops the process after four committed updates, or five hours after launch; the batch allocation has a six-hour bound covering preparation.

## Source and data identity

`scripts/prepare_small_model_profile.py` verifies the frozen source package, copies it, and changes exactly `MODEL_ID` and `MODEL_REVISION` in the copied adapter's `__init__.py`. It creates a new model manifest, verifies downloaded model hashes and tokenizer parity, prepares a new dataset manifest with the existing grader, and requires the retained question IDs and order to match the 14B baseline. Every other runtime source file is byte-identical. The new run receives its own source/data identity. Neither the running 14B release nor official PrimeRL is edited, and no existing run identity is overwritten.

This explicit profile-only model pin avoids relaxing the 14B experiment's model validation. The generated `preparation.json` lists all differences. `PACKAGE_SHA256.json`, the launch scripts, GPU health receipt, topology, allocation and scheduler records make the run inspectable.

## Interpretation

`scripts/run_bounded_profile.py` writes `timing-summary.json` on NFS and, when its run mirror exists, XFS. It extracts generation durations and output-token counts, learner timing, full-update timing and weight-transfer timing. The existing observers publish the ordinary training metrics to Runboard and XFS. Because the supervisor deliberately interrupts a configured longer run, Runboard may show the training process as killed; `timing-summary.json` separately distinguishes successful bounded profiling, early failure and timeout.

These four updates measure bootstrap, not the exact-256 steady phase or checkpoint I/O. Compare measured tokens and response lengths as well as seconds per update: changing model size can change the generated workload. A 1.5B timing result does not establish a proportional speedup for 14B. Do not report a full-run ETA as validated by this short profile.

## Launch

Copy `prepare_small_model_profile.py`, `run_bounded_profile.py` and `profile_small_model_job.sh` into a fresh control directory under the cluster profiling root. The shell script takes that control directory as its only argument. It runs the eight-rank GPU health check, enforces A100 80GB memory, prepares the isolated release and assets, and starts the bounded supervisor. Use the existing pinned PrimeRL environment; do not reinstall it.

Submit with explicit `--account=grad-students --partition=background --qos=background --nodes=1 --exclusive --gres=gpu:a100:8 --nodelist=deep-chungus-7 --cpus-per-task=128 --mem=0 --time=06:00:00 --no-requeue`, after rechecking node availability. This requested node matches the 4+4 topology; a nine-GPU node would require a different configuration.

Validation: wrapper source isolation and tamper detection, successful four-update stop, early launcher failure, and time bound are covered by `tests/test_profile_bounds.py`. These tests do not establish CUDA readiness; the allocated job must pass its hardware and real model startup checks.
