# Imported libraries and study customizations

**Official dependency source files are unchanged.** The study imports a pinned upstream checkout and implements experiment-specific behavior in `src/`. Some integration points use internal APIs, and one trainer factory is temporarily replaced in memory. This page makes those distinctions explicit.

## Pinned source

| Dependency | Revision |
| --- | --- |
| [Official PrimeRL](https://github.com/PrimeIntellect-ai/prime-rl) | v0.9.0, `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1` |
| PrimeRL `uv.lock` SHA-256 | `b55625d807f687401109534bce30bfbb2faf5ba889b246ba521db934d7e1ba4f` |
| Verifiers submodule | `b2e4e8157783b2c0dffc7821044c87f29f1c3ccf` |
| Renderers submodule | `cb8243913702367878427c7a7094b350ea1a8e20` |
| Pydantic-config submodule | `65b15dffba82d4be19efdaf8b2b9705cc1756be8` |
| Prime-envs submodule | `26dafdc9582576975ec576f893be7319028daf51` |

The unused prime-kernels submodule remains uninitialized. No teammate fork is imported. Installation uses `uv sync --frozen`; the bootstrap and runtime constants must agree on the PrimeRL commit. Local source verification during this organization pass found the expected commit, lock digest, submodule revisions and no tracked changes.

The study additionally installs [Runboard](https://github.com/Moe-Zbeeb/runboard) from commit `379e67646391347f50f9e72f1e0226e62c52644f` (version 0.2.0), as a direct dependency in the study's `pyproject.toml`. Bootstrap installs it with `--no-deps` after PrimeRL's frozen sync, without modifying the upstream lockfile or either library's source. Run identity captures the installed Runboard version; the source snapshot binds the exact dependency requirement. This new dependency and tracking source change require matching source/runtime identities on resume.

`runtime/launcher.py::verify_upstream` checks the checkout, submodule state and where the key Python packages were imported from. The run identity records the upstream lockfile and runtime versions. This is a pinned integration, not a promise that a future PrimeRL release will work unchanged.

## Customization inventory

| Study code | Imported interface | What the study changes | Upstream file edited? |
| --- | --- | --- | --- |
| [runtime/build.py](../src/deepseek_study/runtime/build.py) | `RLConfig`, `write_subconfigs` | Resolves the study recipe into trainer/orchestrator/inference configs | No |
| [learning/loss.py](../src/deepseek_study/learning/loss.py) | Custom loss import hook, `LossOutputs` | Uses the explicit clipped GRPO surrogate instead of PrimeRL's default loss | No |
| [learning/advantages.py](../src/deepseek_study/learning/advantages.py) | `Algorithm`, `iter_trainable_traces`, `assign_advantages` | Computes the selected group-advantage normalization; installed on the study's environment instances | No |
| [rollouts/controller.py](../src/deepseek_study/rollouts/controller.py) | `Orchestrator`, dispatcher, `StandardSampler`, training sink, packer, transport | Replaces the dispatcher instance's source with a finite source and drives a complete-cohort loop | No |
| [rollouts/controller.py](../src/deepseek_study/rollouts/controller.py) | `WeightWatcher.apply_policy_update` | Calls weight updates explicitly after generation drains; does not start the automatic watcher loop | No |
| [runtime/trainer.py](../src/deepseek_study/runtime/trainer.py) | `prime_rl.trainer.rl.train.setup_ckpt_manager` | Temporarily replaces this module-level factory in each trainer process with one returning `CheckpointWithRNG`; restores it in `finally` | No; in-memory replacement |
| [runtime/trainer_state.py](../src/deepseek_study/runtime/trainer_state.py) | Official checkpoint manager | Delegates save/load, adds per-rank RNG state and disables upstream cleanup so study retention owns complete bundles | No |
| [runtime/checkpoints.py](../src/deepseek_study/runtime/checkpoints.py) | Official trainer/orchestrator checkpoint output | Commits queue/state metadata and prunes only completed study bundles | No |
| [deepseek_deepscaler/](../src/deepseek_deepscaler/) | Verifiers `Taskset`, `Task`, reward API | Adds the cleaned DeepScaleR taskset and strict math grader | No |
| [dataset/assets.py](../src/deepseek_study/dataset/assets.py) | Transformers tokenizer loading and Renderers | Creates a separate model view with `tokenizer_class=TokenizersBackend`; verifies native token/template parity | No; only generated tokenizer metadata changes |
| [tracking/runboard.py](../src/deepseek_study/tracking/runboard.py) | PrimeRL file-monitor JSONL; Runboard `Run` and saved connection discovery | A separate observer forwards scalar trainer/study metrics and checkpoint completion events; no new trainer callback | No |

The official trainer owns model loading, FSDP, forward/backward, token normalization, accumulation, AdamW, scheduling and weight publication. The study does not patch their implementations. It does change the data supplied to them and the configured objective, so “unmodified library source” must not be interpreted as “all default PrimeRL behavior.”

The custom controller does not run the stock evaluation loop. Adding evaluation settings to an upstream config alone is not an implemented evaluator for this study.

Runboard does not change this boundary. Intermediate evaluation remains disabled, and checkpoint files stay on NFS. Its separate process is excluded from the launcher's training-failure checks and receives final status after training-service shutdown. See the [Runboard guide](runboard.md) for clocks, delivery limits and configuration.

## Upgrades and layout migration

For an upstream upgrade, review the interfaces above before changing the pin. Update the bootstrap/runtime commit together, install the new frozen lock, resolve both hardware profiles, and run the contract suite. Recheck task rendering, sampled-token log-probabilities, sample masks, loss reduction, optimizer-update counting, watcher barriers and checkpoint lifecycle. A live GPU validation is still needed before relying on a changed distributed path.

This source organization changes internal Python module paths, not the scientific recipe. The supported CLI remains `deepseek-study`; the taskset remains `deepseek-deepscaler`. Regenerate resolved configs so their custom-loss path is `deepseek_study.learning.loss.clipped_grpo`. The trainer subprocess is now `deepseek_study.runtime.trainer`; the grader subprocess is `deepseek_study.dataset.worker`.

Existing prepared assets remain valid: the grader source bytes, reward identity and prepared manifest were checked against the previous layout. Source fingerprints and pickle class paths do change. Resume old checkpoints using their original source/environment; do not bypass identity validation or mix source layouts. No study training checkpoints have been created by this task.

Deploy a fresh package/source tree. Preserve existing asset, dependency and output directories deliberately; do not extract over an old tree and leave `data.py` or other retired flat modules behind. `scripts/package.py` builds a complete archive of the new layout with file checksums. The earlier cluster receipts remain historical evidence until the new layout is separately deployed and checked.

## Reviewing future library changes

Any future upstream source patch, additional in-memory override, pin change or tokenizer compatibility change should be recorded here and summarized in the experiment README with the affected interface, reason, behavior and validation. Keep experiment changes in this package where practical so the dependency checkout remains reviewable against the official commit.
