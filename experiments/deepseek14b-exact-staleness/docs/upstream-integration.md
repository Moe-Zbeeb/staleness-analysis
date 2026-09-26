# Imported libraries and study customizations

**Official dependency source files are unchanged.** The study imports a pinned upstream checkout and implements experiment-specific behavior in `src/`. Some integration points use internal APIs, and two trainer factories are temporarily replaced in memory. This page makes those distinctions explicit.

## NCCL startup on PCIe nodes

The launcher calls PrimeRL's `disable_nccl_p2p_if_unavailable` before spawning the trainer and inference processes. It records the resulting `NCCL_P2P_DISABLE` and `NCCL_SHM_DISABLE` values in `run.json`. Explicit paired environment overrides retain the upstream helper's behavior. No upstream source is edited.

The first live 14B diagnostic, job `2144957` on eight A100 80 GB PCIe GPUs, loaded the model but failed during initialization of the separate trainer-to-inference NCCL communicator, before any optimizer update. Isolated probe `2144958` reproduced the late-initialization failure with `Message truncated: received 512 bytes instead of 256`. Both socket-only and peer/shared-memory transports passed when selected before the initial process groups were created. The study therefore applies the existing topology-based fallback at process startup, rather than allowing processes to change transport settings after their initial NCCL groups exist.

`scripts/weight_transfer_probe.py` reproduces the four-trainer/four-inference device split and tests the five-party weight-transfer communicator, without loading model weights. `scripts/readiness.py` checks the corrected startup order before its bounded three-update 14B test and checkpoint recovery. The GRPO objective, data, exact-age queue and model precision are unchanged by this communication fix.

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

The study additionally installs [Runboard](https://github.com/Moe-Zbeeb/runboard) from commit `74b21564d586e43d165d19d2b844ec6cac4deb95` (version 0.2.0), as a direct dependency in the study's `pyproject.toml`. Bootstrap installs it with `--no-deps` after PrimeRL's frozen sync, without modifying PrimeRL's lockfile or source. Runboard's dashboard has been extended in its own repository with generic binned relationship charts; both dashboard asset copies are kept byte-identical. Run identity captures the installed Runboard version; the source snapshot binds the exact dependency requirement. This new dependency and tracking source change require matching source/runtime identities on resume.

`runtime/launcher.py::verify_upstream` checks the checkout, submodule state and where the key Python packages were imported from. The run identity records the upstream lockfile and runtime versions. This is a pinned integration, not a promise that a future PrimeRL release will work unchanged.

## Customization inventory

| Study code | Imported interface | What the study changes | Upstream file edited? |
| --- | --- | --- | --- |
| [runtime/build.py](../src/deepseek_study/runtime/build.py) | `RLConfig`, `write_subconfigs` | Resolves the study recipe into trainer/orchestrator/inference configs | No |
| [runtime/launcher.py](../src/deepseek_study/runtime/launcher.py) | `disable_nccl_p2p_if_unavailable` | Applies the upstream topology-based NCCL transport choice before child processes create communicators | No |
| [learning/loss.py](../src/deepseek_study/learning/loss.py) | Custom loss import hook, `LossOutputs` | Uses the explicit clipped GRPO surrogate instead of PrimeRL's default loss | No |
| [learning/advantages.py](../src/deepseek_study/learning/advantages.py) | `Algorithm`, `iter_trainable_traces`, `assign_advantages` | Computes the selected group-advantage normalization; installed on the study's environment instances | No |
| [rollouts/controller.py](../src/deepseek_study/rollouts/controller.py) | `Orchestrator`, dispatcher, `StandardSampler`, training sink, packer, transport | Replaces the dispatcher instance's source with a finite source and drives a complete-cohort loop | No |
| [rollouts/controller.py](../src/deepseek_study/rollouts/controller.py) | `WeightWatcher.apply_policy_update` | Calls weight updates explicitly after generation drains; does not start the automatic watcher loop | No |
| [rollouts/provenance.py](../src/deepseek_study/rollouts/provenance.py) | `TrainSink.process_batch`, `pending_batch`, `episode_by_trace` | Subclasses the sink, captures sample ownership before upstream clears its buffers, and delegates unchanged batch construction; saves response/question IDs in actual sample order | No |
| [runtime/trainer.py](../src/deepseek_study/runtime/trainer.py) | `prime_rl.trainer.rl.train.setup_ckpt_manager` | Temporarily replaces this module-level factory in each trainer process with one returning `CheckpointWithRNG`; restores it in `finally` | No; in-memory replacement |
| [tracking/tokens.py](../src/deepseek_study/tracking/tokens.py) | `prime_rl.trainer.rl.train.setup_token_exporter` | Replaces the process-local factory with a detached compressed token exporter; enables the upstream export/flush hooks and restores the factory on exit | No; in-memory replacement |
| [runtime/trainer_state.py](../src/deepseek_study/runtime/trainer_state.py) | Official checkpoint manager | Delegates save/load, adds per-rank RNG state and disables upstream cleanup so study retention owns complete bundles | No |
| [runtime/checkpoints.py](../src/deepseek_study/runtime/checkpoints.py) | Official trainer/orchestrator checkpoint output | Commits queue/state metadata and prunes only completed study bundles | No |
| [deepseek_deepscaler/](../src/deepseek_deepscaler/) | Verifiers `Taskset`, `Task`, reward API | Adds the cleaned DeepScaleR taskset and strict math grader | No |
| [dataset/assets.py](../src/deepseek_study/dataset/assets.py) | Transformers tokenizer loading and Renderers | Creates a separate model view with `tokenizer_class=TokenizersBackend`; verifies native token/template parity | No; only generated tokenizer metadata changes |
| [tracking/runboard.py](../src/deepseek_study/tracking/runboard.py) | PrimeRL file-monitor JSONL; Runboard `Run` and saved connection discovery | A separate observer forwards scalar trainer/study metrics and checkpoint completion events; no new trainer callback | No |

The official trainer owns model loading, FSDP, forward/backward, token normalization, accumulation, AdamW, scheduling and weight publication. The study does not patch their implementations. It does change the data supplied to them and the configured objective, so “unmodified library source” must not be interpreted as “all default PrimeRL behavior.”

The custom controller does not run the stock evaluation loop. Adding evaluation settings to an upstream config alone is not an implemented evaluator for this study.

The correctness relaunch separates learner-publication, weight-transfer and checkpoint deadlines. PrimeRL's NCCL broadcast timeout also governs waiting for the inference receiver, and trainer ranks may wait at a collective while generation finishes. `runtime/build.py` therefore sets broadcast and distributed timeouts to cover legitimate phase waits (24 hours in this configuration); the controller still bounds the actual transfer to 30 minutes. The new sink subclass relies on pinned internal object identities. Tests exercise the real upstream sink with interleaved arrivals at both 8 and 512 responses. These changes require revalidation on an upstream upgrade.

Runboard does not change this boundary. Intermediate evaluation remains disabled, and checkpoint files stay on NFS. Its separate process is excluded from the launcher's training-failure checks and receives final status after training-service shutdown. See the [Runboard guide](runboard.md) for clocks, delivery limits and configuration.

## Upgrades and layout migration

For an upstream upgrade, review the interfaces above before changing the pin. Update the bootstrap/runtime commit together, install the new frozen lock, resolve both hardware profiles, and run the contract suite. Recheck task rendering, sampled-token log-probabilities, sample masks, loss reduction, optimizer-update counting, watcher barriers and checkpoint lifecycle. A live GPU validation is still needed before relying on a changed distributed path.

This source organization changes internal Python module paths, not the scientific recipe. The supported CLI remains `deepseek-study`; the taskset remains `deepseek-deepscaler`. Regenerate resolved configs so their custom-loss path is `deepseek_study.learning.loss.clipped_grpo`. The trainer subprocess is now `deepseek_study.runtime.trainer`; the grader subprocess is `deepseek_study.dataset.worker`.

The original layout-only reorganization preserved prepared assets. The later structure/case grader correction changes reward identity and requires a new prepared manifest; the corrected manifest retains the same question selection. Source fingerprints and pickle class paths also change. Resume old checkpoints using their original source/environment; do not bypass identity validation or mix source layouts or reward policies. The organization pass created no training checkpoints; the later [readiness test](readiness-test.md) uses separate diagnostic recovery bundles.

Deploy a fresh package/source tree. Preserve existing asset, dependency and output directories deliberately; do not extract over an old tree and leave `data.py` or other retired flat modules behind. `scripts/package.py` builds a complete archive of the new layout with file checksums. The organized release `releases/paper-metrics-20260926` was separately deployed and checked in CPU-only job `2144955`; the preceding hardware receipts still describe component checks, not end-to-end model training.

## Reviewing future library changes

The standalone `scripts/profile_trainer.py` diagnostic adds process-local timing wrappers around the imported trainer's forward, loss, optimizer and backward calls, replaces its fake loader with archived microbatches, and times the study token exporter. It operates only in a disposable diagnostic process, with policy publication and checkpoint saving disabled. Production does not import these hooks; upstream source is unchanged. See the [profiling record](profiling.md) for measurement caveats and cluster results.

Any future upstream source patch, additional in-memory override, pin change or tokenizer compatibility change should be recorded here and summarized in the experiment README with the affected interface, reason, behavior and validation. Keep experiment changes in this package where practical so the dependency checkout remains reviewable against the official commit.

## Paper diagnostics

The [paper metric inventory](paper-metrics.md) describes the additional export factory, CPU aggregation, XFS mirror and offline evaluation import. Capturing entropy uses the existing forward output; the loss, optimizer and update clock are unchanged. This adds I/O and CPU overhead, not a loss term. Model checkpoints remain on NFS.

The token-contribution update adds detached loss diagnostics and schema-2 raw masks (`surrogate_clipped`, `zero_policy_signal`). The GRPO objective and its autograd path are unchanged. PrimeRL source remains unmodified; the masks are computed by our loss/export adapters and globally counted by the existing paper observer.

Live diagnostic `2144959` generated and graded 32 responses, totaling 186,397 response tokens, and reached backward execution. It then exposed a device mismatch in the token exporter before the first optimizer update: PrimeRL passes the original CPU microbatch alongside GPU model outputs to its export hook. The exporter now moves only the detached mask, behavior log probabilities and advantages to the model-output device before calculating contribution masks. This keeps the mask arithmetic on the same device as the GRPO calculation and leaves autograd unchanged. A CPU/GPU parameterized regression checks the saved masks against actual gradients with CPU rollout metadata and GPU outputs; the CUDA case runs inside the allocated readiness job.
