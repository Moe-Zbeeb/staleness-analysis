# September 27 hardening and validation

All training is stopped. This release fixes failures observed in the saved logs and adds regression coverage. It has not completed a GPU training pilot and cannot guarantee that future generations, hardware or storage will never fail. No training was launched while making these changes; the health monitor remains paused.

## Observed failures

| Run | Last state | Cause and consequence |
| --- | --- | --- |
| DeepSeek 14B, job 2144963 | Failed after 7 committed updates | The reference `6:00` became complex infinity in the mathematical parser; verification raised `ValueError` in update 8. |
| Qwen3-1.7B, job 2145033 | Failed after 7 committed updates | Same clock reference and grading failure. |
| DeepSeek 1.5B, job 2145032 | Preempted before its first committed update | Scheduler interruption, not a diagnosed model or loss error. |
| Qwen2.5 base 3B, job 2145034 | Cancelled after 77 committed updates | Every recorded response was rejected for missing `</think>`, a format not opened by the base model's native prompt. |

None reached the first checkpoint at update 100. The 14B continuation correctly refused to restart without a complete checkpoint. Preemption can still lose work since the last completed checkpoint; the requested 100-update cadence and no-automatic-restart policy remain unchanged.

## Changes

- Grading uses the raw question to distinguish clock times from ratios. Strict valid `HH:MM` comparison replaces mathematical division for clock references. Native reasoning models still require a closed reasoning section; base Qwen2.5 accepts a plain boxed final answer. The setting is validated against the pinned model and recorded in configuration, protocol and data identity.
- Preparation checks every reference for defined parsed values and self-verification, records unsupported references, and rejects a manifest if the grading contract changes during preparation. Original question context is kept separate from appended prompt instructions.
- Recognized malformed predictions receive zero reward with an explicit reason. Unexpected errors and timeouts retain bounded retries and stop the cohort if persistent. Worker reports now include the actual exception message and traceback.
- Packing and dispatch have a separate 30-minute deadline. A timed-out dispatch cannot advance the controller's policy/update clock or silently retry the cohort.
- Shutdown tracks only child process groups started by this launcher, bounds waits after termination, handles signals during child startup, and preserves the original error. Final status records signal, child exit codes and Slurm identity. Status stays immutable for NFS/XFS mirroring; late observer warnings stay in the launcher log.
- GRPO rejects nonfinite trainable current or behavior log probabilities before backward. Existing finite behavior, including finite numerical underflow and masked prompt tokens, is preserved.
- Run metadata and Runboard tags identify the actual model, including small-model runs.

Official PrimeRL remains pinned at `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1`, with no tracked source modifications. The GRPO formula, advantages, masking, clipping, optimizer settings, exact-age schedule, cohort size, context limits, GPU topology and checkpoint cadence are unchanged. Reward interpretation and dataset membership have changed and are explicitly versioned.

## Validation evidence

Local validation: **298 tests passed, one CUDA-only test skipped**. Ruff lint and the changed-file formatting checks passed. Coverage includes exact-age scheduling, bounded dispatch, nonfinite loss inputs, source/data identity, signal/cleanup races, taskset-to-grader context, Runboard metadata and the observed grading failures.

Full reference preparation completed for all 37,713 locked rows: **37,696 included, 17 excluded**. Seven exclusions are new: multiple clock times, seconds/fractional minutes, an explicit minute-second duration, and a match score requiring distinct structured comparators. The original labels and source parquet are unchanged. [The receipt](../diagnostics/hardening-20260927.json) lists the exact question IDs, original references, exclusion reasons and hashes.

The new manifest body hash is `c86db656d027e582ad712f8c100f17a5f6c75b54fea7e795f4e1fca6e3a5c90f`. The grader source hash is `1efc5e898c499e3102c506efc3f459ab55ff1b4839b112c1a59edf132ee82235`.

Saved failure responses were reconstructed from original sampled token IDs using each model's native tokenizer, rather than display-rendered text. The actual 14B `6:00\ \text{pm}` and Qwen3 `6:00` answers both return reward 1 on the first isolated-worker attempt. Unqualified clock references verify only the clock-face value, not AM/PM; see [grading policy v3](grading-v3.md).

The saved first 3B warmup cohort was regraded offline, preserving explicit sample-to-response/question mappings and truncation flags. **76/512 responses are correct, versus zero before. 30/64 groups have mixed rewards, giving 240 responses nonzero centered advantages.** All 512 completed on the first attempt. This demonstrates a restored reward signal on that saved batch; it does not measure later learning or benchmark accuracy.

## Deployment and next validation

The new cluster release is `releases/correctness-v3-20260927` under `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study`. It includes the v3 data manifest and hash inventory. Old v2 releases, output snapshots and launch controls must remain unchanged as historical evidence.

New runs require newly generated configurations/manifests and fresh output directories. The `init` command now selects `assets/train-manifest-v3.json`. Historical `exact256-*-v2.json` files and small-model launch controls refer to the old source/data contract and must not be reused. Prepare each small-model adapter against the new frozen release; its tokenizer/model and reasoning format must match. Old checkpoints cannot be relabeled as v3.

Before another full study, run a bounded k=8 pilot through bootstrap, exact-age updates, a complete checkpoint and resume. A 40-update diagnostic with a checkpoint at update 20 would exercise those paths and the end of the 30-update LR warm-up. That diagnostic must use its own explicit configuration and output directory. A subsequent full study starts fresh with its intended budget and k; do not resume a k=8 pilot into k=256. No pilot has been submitted by this change.
