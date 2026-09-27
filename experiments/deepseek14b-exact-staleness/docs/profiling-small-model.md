# Bounded 1.5B timing profile

This job measures the first four complete updates of `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, pinned to Hugging Face revision `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`. The user chose 1.5B only; no 3B job is submitted.

The baseline is the frozen configuration and source used by 14B job `2144963`: `releases/correctness-v2-20260926`, source commit `5ad56dedbd4bfd3a4e3bb2f39ac4e56d0de274d8`. The profiling job uses eight A100 GPUs, with four inference GPUs and four trainer GPUs. After the requested eight-GPU node failed CUDA initialization, the user explicitly accepted hardware differences and the move to eight GPUs on `deep-chungus-5`. This job keeps the node exclusive but requests eight of its nine GPUs to preserve the 4+4 geometry; one physical GPU is unused. Background partition/QoS is selected as requested; preemption can interrupt the measurement. Automatic requeue is disabled to prevent duplicate writers.

All training settings stay identical: 64 prompts × 8 responses, 2,048 prompt and 8,192 response token limits, 1e-6 learning rate with 30-update warmup, seed 42, zero weight decay, centered advantages, clipped GRPO, the same logging, native tokenizer preparation, and no online evaluation. Model, prepared assets, data manifest and output paths are isolated. The 1,000-update budget, exact lag 256 and checkpoint interval 100 remain configured. An external supervisor stops the process after four committed updates, or five hours after launch; the batch allocation has a six-hour bound covering preparation.

## Source and data identity

`scripts/prepare_small_model_profile.py` verifies the frozen source package, copies it, and changes exactly `MODEL_ID` and `MODEL_REVISION` in the copied adapter's `__init__.py`. It creates a new model manifest, verifies downloaded model hashes and tokenizer parity, prepares a new dataset manifest with the existing grader, and requires the retained question IDs and order to match the 14B baseline. Every other runtime source file is byte-identical. The new run receives its own source/data identity. Neither the running 14B release nor official PrimeRL is edited, and no existing run identity is overwritten.

This explicit profile-only model pin avoids relaxing the 14B experiment's model validation. The generated `preparation.json` lists all differences. `PACKAGE_SHA256.json`, the launch scripts, GPU health receipt, topology, allocation and scheduler records make the run inspectable.

## Interpretation

`scripts/run_bounded_profile.py` writes `timing-summary.json` on NFS and, when its run mirror exists, XFS. It extracts generation durations and output-token counts, learner timing, full-update timing and weight-transfer timing. The existing observers publish the ordinary training metrics to Runboard and XFS. Because the supervisor deliberately interrupts a configured longer run, Runboard may show the training process as killed; `timing-summary.json` separately distinguishes successful bounded profiling, early failure and timeout.

These four updates measure bootstrap, not the exact-256 steady phase or checkpoint I/O. Compare measured tokens and response lengths as well as seconds per update: changing model size can change the generated workload. Hardware and CPU-count differences are recorded, so this is not a controlled model-size-only experiment. A 1.5B timing result does not establish a proportional speedup for 14B. Do not report a full-run ETA as validated by this short profile.

## Launch

Copy `prepare_small_model_profile.py`, `run_bounded_profile.py` and `profile_small_model_job.sh` into a fresh control directory under the cluster profiling root. The shell script takes that control directory as its only argument. It runs the eight-rank GPU health check with a five-minute bound, requires A100 GPUs with at least 40GB nominal memory, prepares the isolated release and assets, and starts the bounded supervisor. Use the existing pinned PrimeRL environment; do not reinstall it.

Submit with explicit `--account=grad-students --partition=background --qos=background --nodes=1 --exclusive --gres=gpu:a100:8 --nodelist=deep-chungus-5 --cpus-per-task=96 --mem=0 --time=06:00:00 --no-requeue`, after rechecking node availability. Using eight of nine GPUs is an explicit exception for this comparison, not a new default allocation policy.

The first attempt, job `2145012` on `deep-chungus-7`, failed its startup probe after 151 seconds with CUDA reporting `device=7, num_gpus=7`. It completed no model preparation or training. Its logs and original submitted scripts remain under `profiling/20260927-deepseek15b`; the replacement uses a separate control directory.

On node 5, Slurm's exclusive reservation accounts for all nine GPUs despite the eight-GPU request. Job `2145013` failed its preflight device-count guard after 13 seconds, before training. The corrected wrapper selects eight entries from Slurm's actual `CUDA_VISIBLE_DEVICES` list before starting any CUDA process and records the complete allocation and unused device separately. It never invents physical device indices or accesses a GPU outside the allocation.

Replacement job **2145014** is on `deep-chungus-5`, account `grad-students`, partition/QoS `background`, 96 CPU slots, all node memory, six-hour allocation and no requeue. Its eight participating **A100 PCIe 40GB** GPUs passed NCCL all-reduce (sum 36), BF16 backward, Flash Attention backward and vLLM RMSNorm checks. Devices 0–3 serve inference; 4–7 train; device 8 is unused for this comparison. The 14B production job remains unchanged on its separate eight-A100-80GB node.

The launch directory is `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/profiling/20260927-deepseek15b-node5-v2`. The isolated source, assets, configuration and training output live beneath its `work` directory. The timing summary will be written to `work/timing-summary.json`; the metric mirror is `/mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics/deepseek15b-exact256-timing-2145014`. Its Runboard name is `deepseek15b-exact256-timing-2145014` once the launcher starts. The [submission and hardware receipt](../diagnostics/profiling-20260927-small-model.json) records the observed stage; submission and hardware checks alone do not prove completed training updates.

Validation: wrapper source isolation and tamper detection, successful four-update stop, early launcher failure, and time bound are covered by `tests/test_profile_bounds.py`. These tests do not establish CUDA readiness; the allocated job must pass its hardware and real model startup checks.
