# Exact-staleness implementation

The public configuration accepts one explicit nonnegative integer lag. A fresh run has lag on-policy bootstrap updates, followed by the exact-lag phase. The total horizon includes bootstrap. Each complete cohort causes one optimizer update after all packed microbatches accumulate.

`recipe.py` writes the reviewed baseline; `config.py` rejects incompatible horizons, layouts, deadlines and request concurrency. `build.py` resolves the pinned official PrimeRL configuration. `queue.py` selects cohorts by behavior-policy version and overlaps generation with training; every response is consumed once, and the tail drains without unused generation. The finite source addresses the official cyclic sampler by target consumption cohort, preserving prompt assignments across lag choices.

`controller.py` composes the official orchestrator, dispatcher, training sink, packer and transport. It does not start the automatic newest-weight watcher. It explicitly applies the next policy only after the finite generation cohort completes. It verifies per-response start/end versions, payload hashes and exact consumption age. Raw token/log-probability payloads and grading evidence are journaled.

`algorithm.py` assigns explicit group advantages. `loss.py` implements the clipped GRPO surrogate from original behavior log-probabilities, with token masks, distinct clipping metrics and a nonfinite-value guard. The official trainer supplies global token normalization, accumulation, optimizer updates and distributed training.

`assets.py` verifies source model/data identity and native-tokenizer parity. `data.py` records deterministic prompt/reference exclusions without modifying the source dataset. `rewards.py` defines the pinned strict-box policy; `grading.py` manages persistent isolated workers with deadlines and retry-on-identical-input semantics. `deepseek_deepscaler` integrates these with the official taskset API.

`identity.py` creates an immutable source snapshot for child processes and binds the source, upstream dependency lockfile, runtime and prepared data. `trainer_state.py` adds rank-local RNG save/restore around the official checkpoint manager through the study entrypoint; upstream source is unmodified. `checkpoints.py` commits the full trainer/sampler/queue bundle and prunes only completed run checkpoints. `audit.py` validates the requested lag and update/cohort accounting, including resumed output directories.

The extension uses pinned internal PrimeRL interfaces; an upstream upgrade requires rerunning contract tests. CPU validation does not establish real GPU memory fit, NCCL transfer, live inference numerical parity or bitwise inference recovery. Automatic benchmark evaluation and cross-hardware resume are not implemented.
