# Qwen2.5-3B full exact-256 study on high priority

Current attempt: **2145148**, high-priority node 7, TensorBoard only, 1,000 updates at k=256. See the final submission section below for the active controls and source; earlier attempts are retained as diagnostic history.

The user explicitly selected the full 1,000-update k=256 study after the v3 hardening review. This starts the pinned base `Qwen/Qwen2.5-3B` revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` from its original weights. It does not resume the old zero-reward run or use a bounded timing supervisor.

Job **2145127** was submitted at **2026-09-27 12:39:11 UTC** to account `grad-students`, partition/QoS `high-priority`, on `deep-chungus-7`. Slurm confirmed an exclusive eight-A100 allocation, 128 CPU slots, a 45-day wall limit and no automatic requeue. The node's previously unusable eighth device may be excluded under the user's existing seven-healthy-GPU authorization. Preparation selects four trainers plus four inference workers if all eight pass, or four plus three if exactly seven pass. Occupied devices, wrong memory capacity, duplicate GPU identities or fewer than seven healthy GPUs abort the launch.

## Protocol and preparation

The run keeps 64 prompts × 8 responses, a 2,048-token prompt cap, an 8,192-token response cap, seed 42, LR 1e-6, clip epsilon 0.2, zero weight decay and the existing GRPO objective. The first 256 updates build history; update 257 is the first exact-age-256 update. Checkpoints are saved every 100 updates and at completion on NFS. Metrics also go to XFS and Runboard. Evaluation remains offline only.

The driver verifies all launch-script hashes before checking GPUs, clones the frozen v3 adapter, verifies the cached model weights against their pin, creates a fresh native tokenizer view and prepares a new grading manifest. Base Qwen uses `reasoning_required=false`. The selected question IDs and order must match the v3 14B manifest. The old profile's model files are reused read-only; none of its training updates, source identity or old grading manifest are reused.

A fresh-study mode in `prepare_full_run.py` captures the new source/data/runtime identity directly. The launcher then repeats device checks and verifies NCCL, BF16 backward, Flash Attention backward and vLLM RMSNorm before starting training. This preserves the scientific protocol while replacing the historical-profile dependency of the old launch preparer. Local validation passed **325 tests**, with one CUDA-only test skipped.

## Paths and evidence

- Launch controls: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/qwen25-3b-full-hp-v3-20260927`.
- Slurm logs and exact submission receipt: the sibling `qwen25-3b-full-hp-v3-20260927-logs` directory.
- Fresh model adapter, tokenizer view and manifest: `work/` inside the launch controls.
- Output: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/outputs/qwen25-3b-exact256-80gb-seed42-v3`.
- Code for the new launch path: commit `3dff15e`; scientific source derives from the frozen `correctness-v3-20260927` release.
- [Submission and validation receipt](../../diagnostics/qwen25-3b-full-hp-v3-20260927.json).

At the initial check, Slurm was running the preparation phase and no training update had completed. This is a launch record rather than a promise of completion or a timing estimate. Subsequent startup evidence is recorded below.

## Startup deadline correction

Initial job 2145127 stopped safely before source/model preparation or training: every per-device subprocess exceeded the old 120-second health-check deadline. Bounded diagnostic job **2145128** completed in 182 seconds. Seven A100-80GB cards passed the memory guard and BF16 backward checks. Their subprocesses took 163–176 seconds in total; instrumented PyTorch import took about 42 seconds and CUDA initialization completed around 142 seconds after the Python payload began. Device 7 raised `No CUDA GPUs are available` and remains excluded under the authorized fallback.

The new full-launch startup allowance is 300 seconds per device plus 60 seconds for the parent probe, and 900 seconds for the combined collective/library health process. The NCCL operation deadline, memory threshold, BF16/Flash Attention/vLLM checks and all training settings remain unchanged. Stage timings and timeout stderr are now retained for diagnosis. The failed controls and diagnostic logs remain intact. [Diagnostic receipt](../../diagnostics/qwen25-hp-startup-diagnostic-20260927.json).

## Corrected full-run submission

Replacement job **2145130** was submitted at **2026-09-27 12:53:26 UTC**, using launch-control commit `fdd73bb`. Scheduler fields again confirm `grad-students` / `high-priority` / `high-priority`, exclusive eight-GPU node 7, 128 CPU slots, 45 days and no requeue. The control directory is `launches/qwen25-3b-full-hp-v3r2-20260927`, with logs in its sibling `qwen25-3b-full-hp-v3r2-20260927-logs` directory. The training output name remains `qwen25-3b-exact256-80gb-seed42-v3`; the first failed attempt never created it or performed any update. The additional startup changes passed 43 targeted tests.

## User-requested tracking change

Job **2145130** was canceled at the user's request to switch to TensorBoard only. Slurm confirms cancellation after 7m33s; the training output directory had not been created, and zero training updates were committed. Preparation completed, but training did not begin. Its launch controls and preparation artifacts are retained unchanged. The replacement uses a new frozen source/control directory, with Runboard disabled and TensorBoard event logs mirrored to XFS.

## TensorBoard-only replacement

Job **2145141** was submitted at **2026-09-27 13:14:43 UTC** on `deep-chungus-9`, which became idle and previously passed eight-A100-80GB checks. This job requires all eight cards to pass current allocation health checks, with four trainers and four inference workers. Scheduler admission confirms high-priority partition/QoS, the `grad-students` account, exclusive full-node allocation, 128 CPUs, a 45-day wall limit and no requeue.

The frozen source is `releases/tensorboard-v4-20260927`, from commit `d6e8747`. All 17 files implementing the loss/advantages, rollout schedule, dataset/grader, task adapter, config and recipe were compared byte-for-byte with the corrected v3 release and matched. The grading policy remains v3. This change replaces tracking with TensorBoard and gives this fresh full-node run four inference workers.

Controls are `launches/qwen25-3b-full-hp-tb-20260927`; logs are in the sibling `-logs` directory. Output is `outputs/qwen25-3b-exact256-80gb-seed42-v3-tb`. TensorBoard events go under its `tensorboard/` directory and the corresponding XFS metric mirror. The launcher never starts Runboard. See the [TensorBoard guide](../tensorboard.md) and [submission receipt](../../diagnostics/qwen25-3b-tensorboard-full-hp-20260927.json).

The full local suite passed 348 tests with one CUDA-only skip. After the final bounded-memory replay implementation, 35 focused logging, mirror, paper-metric and launcher tests passed, with the same CUDA-only skip. Cluster startup verification is recorded in the receipt as it becomes available.

Node 9 acquired an unrelated CPU-only job before the exclusive allocation could start. Job 2145141 was canceled while pending, with no output or training. Replacement **2145142** was submitted at **13:18:31 UTC** on idle node 7 using the same frozen TensorBoard source and the previously authorized seven-healthy-GPU fallback. Its fresh controls are `launches/qwen25-3b-full-hp-tb-node7-20260927`. The eight configured GPUs are allocated exclusively; if the known unusable eighth card fails again, all seven healthy cards are assigned once, four to training and three to inference. No unrelated job was altered.

Job 2145142 stopped safely before training after 5m13s: all eight health-probe children timed out during `import torch`, before CUDA was reached. The single-process diagnostic 2145146 completed in 2m19s. Its CPU-only Torch import took 12.37s, subsequent parallel imports took 1.59–1.78s, and GPUs 0–6 passed A100-80GB/BF16 checks in about 102.5s. GPU 7 retained its known `No CUDA GPUs are available` error. These measurements support a bounded single import before the parallel checks, but do not establish shared-storage contention as the original stall's cause. An earlier diagnostic wrapper path error (2145145) was corrected before these measurements.

The launch path now imports Torch once in a CPU-only subprocess before each parallel GPU probe. This uses the same interpreter, hides GPUs only for that child, has a 180-second deadline and records traceback stacks every 30 seconds. Failed imports block the GPU fan-out and leave an exclusive diagnostic receipt. GPU deadlines remain 300/360 seconds, collective checks remain 900 seconds, and the training recipe is unchanged. This startup change passed 45 targeted tests before deployment.

Current full job **2145148** was submitted at **13:37:59 UTC** on high-priority node 7, using frozen release `tensorboard-v4r2-20260927` from commit `ca49388`. Fresh controls are `launches/qwen25-3b-full-hp-tb-warm-20260927`. The output remains `qwen25-3b-exact256-80gb-seed42-v3-tb` because previous attempts never created it. The final launch/logging regression selection passed 66 tests. `current` now points to this new release; earlier source and controls remain preserved.

Latest startup snapshot for 2145148: Slurm RUNNING at 15m51s, with no committed optimizer update, training output directory or TensorBoard events yet. Both GPU probe rounds passed seven cards; the source, model, tokenizer and 37,696-question membership checks completed. The final collective-health subprocess remained within its 900-second deadline. Its seven workers showed filesystem/RPC waits (`rpc_wait_bit_killable`) while read counters increased. This is not a demonstrated GRPO, TensorBoard or NCCL numerical failure, and the exact blocked file is not established. The job was left running under its existing startup deadline; successful training is not yet confirmed.
