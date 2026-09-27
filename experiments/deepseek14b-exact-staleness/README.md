# DeepSeek 14B / cleaned DeepScaleR / exact staleness

One run at a time, with the exact nonnegative integer `k` you request. This package composes official PrimeRL v0.9.0 at `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1`; it does not import the teammate fork or edit upstream files.

**Status (September 27, 2026): TensorBoard-only full 3B job 2145148 is delayed in collective-health startup on high-priority node 7 (zero training updates at the latest check); the other full studies remain stopped.** The user selected a fresh 1,000-update k=256 run with the corrected v3 grader. See the [3B launch record](docs/runs/qwen25-3b-full-hp-v3.md) for the initial startup timeout, measured correction, new job and verified startup state. The [hardening report](docs/hardening-20260927.md) documents the preceding fixes and offline validation. Historical 14B/1.7B grading failures, 1.5B preemption and stopped 3B outputs are preserved.

The preceding bounded readiness job `2144960` passed (exit `0:0`, 54m 45s), including all 129 cluster tests. The real 14B model completed three updates with ages 0, 1, 1, then resumed checkpoint 2 and reexecuted update 3 using the original queued data. Responses reached 8,192 tokens; trainer peak memory reached about 72 GiB after restart. Checkpoints are on NFS and metric evidence is mirrored to XFS. See the [readiness report](docs/readiness-test.md) and [validation evidence](diagnostics/cluster-validation.json).

The diagnostic found and fixed two integration bugs: NCCL transport settings were applied too late, and the token exporter mixed CPU rollout tensors with GPU model outputs. Both fixes are in this package. The official PrimeRL checkout, GRPO objective, rollout scheduler and production configuration remain unchanged.

## Review and organization

The [throughput profiling record](docs/profiling.md) documents the measured production bottlenecks, bounded diagnostic jobs, reusable profilers and their validation limits. Profiling tools do not modify the production run or imported library files.

The historical [1.5B timing profile](docs/profiling-small-model.md) uses an isolated copy of the frozen 14B adapter with only its model pin changed. It preserves the training settings and stops after four bootstrap updates. Official PrimeRL and the frozen 14B source remain unchanged.

The [Qwen2.5-3B timing profile](docs/profiling-qwen3b.md) uses the requested base checkpoint and native tokenizer, with an explicitly authorized seven-GPU fallback on node 7. Its isolated adapter changes are documented separately.

The [Qwen3-1.7B timing profile](docs/profiling-qwen3-1p7b.md) uses native thinking mode on node 4 with four trainers and five inference workers. It preserves the learner settings and the same four-update bound.

Start with the [architecture](docs/architecture.md) for the directory tree and execution flow, then the [review guide](docs/review-guide.md) for the exact-k contract and checks.

| Concern | Location | Edit here when… |
| --- | --- | --- |
| Baseline values | [recipe.py](src/deepseek_study/recipe.py) | Choosing the explicit settings for a run |
| Supported settings | [config.py](src/deepseek_study/config.py) | Defining or validating a scientific setting |
| Command interface | [cli.py](src/deepseek_study/cli.py) | Working on setup, build, check, run or audit commands |
| Advantages and loss | [learning/](src/deepseek_study/learning/) | Changing the learning algorithm |
| Exact-age cohorts and audit | [rollouts/](src/deepseek_study/rollouts/) | Changing generation, queueing or consumption |
| Assets, preparation and grading | [dataset/](src/deepseek_study/dataset/) | Working on the dataset or reward policy |
| PrimeRL configuration, processes and recovery | [runtime/](src/deepseek_study/runtime/) | Working on the library integration or checkpoint lifecycle |
| TensorBoard metrics and event archives | [tracking/](src/deepseek_study/tracking/) | Observing existing logs without changing training |
| Taskset plugin | [deepseek_deepscaler/](src/deepseek_deepscaler/) | Connecting prepared questions and rewards to Verifiers |
| Installation and health checks | [scripts/](scripts/) | Preparing dependencies, packaging or checking hardware |
| Tests and evidence | [tests/](tests/), [diagnostics/](diagnostics/) | Reviewing validated behavior and limits |

Configuration templates/schema live in `configs/`; pinned model hashes are in `manifests/`. Preparation writes the question-selection record to `assets/train-manifest-v3.json`. Weights, raw datasets, generated manifests, environments, caches, run outputs and job logs are excluded from Git.

The [staleness research audit](docs/staleness-research-audit.md) separates exact age, policy difference, learning quality and throughput. It maps the relevant knobs, differences from the cited paper, current measurement gaps and a proposed research protocol. It does not launch experiments or change the baseline.

## Imported libraries: what we change

**No tracked source files in official PrimeRL or its imported submodules are modified.** We pin PrimeRL v0.9.0 at `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1` and install its frozen dependency lock. The launcher rejects tracked changes or a different dependency revision.

We do customize behavior through our package: a configured loss, a group-advantage algorithm, a finite rollout source and explicit weight synchronization. `ProvenanceTrainSink` subclasses the pinned training sink to attach response/question identities to its existing sample order. In each trainer process, `runtime/trainer.py` temporarily replaces PrimeRL's checkpoint-manager and token-exporter factories with our RNG-saving and compressed diagnostic adapters. Both factories are restored on exit. These are in-memory overrides; upstream source is unchanged. A separate prepared model view changes only tokenizer-class metadata for compatibility; the original model files remain untouched.

The [integration guide](docs/upstream-integration.md) lists every customization, the dependency pins, and what must be reviewed when upgrading the library. These internal interfaces are version-sensitive; the package does not automatically follow upstream changes.

## Source-layout update

The original module reorganization moved implementation files into four concern-specific subpackages without changing scientific behavior; `tracking/` contains independent TensorBoard and paper-metric observers. The original reorganization changed the resolved custom-loss import path to `deepseek_study.learning.loss.clipped_grpo`. The original reorganization preserved the advantage, loss, queue and reward implementation bytes; the grader identity and prepared dataset manifest are unchanged. Subsequent baseline changes to weight decay and checkpoint cadence are described below. The `track` command now exports TensorBoard events; Runboard is an optional historical dependency and is never started by the current launcher.

Regenerate resolved configs from the study JSON using `deepseek-study build`; do not reuse old resolved files with the flat loss import path. Source fingerprints and pickled module paths changed, so old source snapshots/checkpoints require their original code. Use a fresh source deployment for this layout; do not overlay it onto an old source tree and leave obsolete modules behind. [Upgrade and deployment notes](docs/upstream-integration.md#upgrades-and-layout-migration).

## Training contract

`theta_t` is the model after `t` optimizer updates. After the initial bootstrap, training `theta_t` consumes responses generated entirely by `theta_(t-k)`. Microbatches accumulate into one update; responses are trained on once. For `k=32`, updating `theta_100` uses responses from `theta_68`.

A fresh run first performs `k` explicitly labeled on-policy bootstrap updates while generating distinct future cohorts. Update `k+1` is the first exact-k update. With `k=0`, every update is on-policy. The total budget includes bootstrap: a 1,000-update run at k=32 has 32 bootstrap and 968 exact-k updates. A budget that leaves no exact-k updates is rejected.

Generation uses frozen current inference weights and queues its original responses until the prescribed update. Training and generation overlap where possible. Inference weights change only after the complete cohort finishes. Tail generation stops early enough that no extra cohorts remain after the final update. Cohort prompts are indexed by their intended consumption update, so changing k does not change the assigned prompt stream.

## Historical first run: exact 256

The stopped v2 run is recorded in [configs/exact256-80gb-seed42-v2.json](configs/exact256-80gb-seed42-v2.json): exact lag 256, the 80 GB profile, seed 42 and 1,000 total optimizer updates. This means 256 on-policy bootstrap updates followed by 744 exact-256 updates; update 257 first consumes a queued cohort generated by `theta_0` while training `theta_256`. This historical configuration refers to its original grading/data contract. Create a new configuration with `init` for v3; do not reuse its old manifest.

See the [relaunch record](docs/runs/exact256-v2.md) for corrected deadlines, source and reward identities, and cluster paths. The model, optimizer, loss and exact-age schedule are unchanged; the corrected reward policy requires a fresh run.

## Baseline recipe

The `init` command writes a complete configuration. Its defaults are 64 prompts × 8 responses = 512 responses per update, a 2,048-token rendered-prompt cap, 8,192 response tokens, seed 42, and 1,000 total updates. Sampling uses temperature 1, top-p 1, disabled top-k/min-p, and neutral penalties.

The loss is clipped GRPO with epsilon 0.2, reward-minus-group-mean advantages without standard-deviation normalization, and a global token mean. Reference KL and entropy coefficients are zero. Zero-advantage groups remain in the denominator. An all-zero cohort still executes AdamW; existing optimizer momentum may move the policy even with weight decay disabled.

AdamW uses LR 1e-6, betas 0.9/0.999, epsilon 1e-8, weight decay 0 and gradient norm clipping at 1. Disabling weight decay removes parameter shrinkage independent of the GRPO loss gradient; the exact-age queue and GRPO objective are unchanged. LR warms up over 30 optimizer updates, then remains constant. This LR warm-up is distinct from the k-update bootstrap. Full-model training uses BF16 compute, FP32 optimization/reduction, activation checkpointing, and no quantization. Compilation and prefix caching are initially disabled.

The baseline and configuration template previously used weight decay 0.01. Existing run JSON files retain their explicit settings: set `weight_decay` to `0.0` and rebuild resolved configs before a new run, or create a new configuration with `init`. This is a scientific configuration change, so it must not be applied while resuming an older recipe. The current cluster release includes zero weight decay.

| Profile | Trainer / inference GPUs | Request concurrency | Active sequences per inference replica | Activation CPU offload |
|---|---:|---:|---:|---|
| `80gb` | 4 / 4, inference TP=1 | 64 | 16 | Off |
| `40gb` | 7 / 1, inference TP=1 | 8 | 2 | On |

The 40 GB profile needs at least eight outstanding requests to satisfy PrimeRL's group-size constraint; vLLM still limits active sequences to two. These are configurations for GPU validation, not proven memory-fit or throughput results. The defaults target an FA2-compatible runtime; select the matching attention backend when using other hardware. No Slurm partition or QoS is guessed by the launcher.

## Installation and one-run configuration

Run these commands from the project root. Bootstrap creates its own pinned environment; do not copy a macOS virtual environment onto Linux.

```bash
uv run --no-project --python 3.12 scripts/bootstrap.py --gpu --attention fa2
vendor/prime-rl/.venv/bin/deepseek-study init configs/run.json --lag 32 --profile 80gb
```

Here 32 is an example of an explicitly selected run. Use the requested k instead. `init` refuses to overwrite an existing configuration. `--seed`, `--max-steps` and `--root` override their defaults. Review paths in the generated JSON before launch. Short diagnostics may also need an explicitly shorter LR warm-up; the configuration does not silently change it.

For CPU development, omit `--gpu --attention fa2` from bootstrap.

## Model and data preparation

The original model is `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` at revision `1df8507178afcc1bef68cd8c393f61a886323761`. Model files are checked against `manifests/model.json`. The prepared view links the verified files and changes only tokenizer-class metadata to preserve native tokenization under the pinned Transformers version.

```bash
vendor/prime-rl/.venv/bin/deepseek-study prepare \
  --model /mnt/nfs/home/mohamadzbib/projects/models/exact-age-14b/deepseek-r1-distill-qwen-14b \
  --dataset /mnt/nfs/home/mohamadzbib/projects/exact-age-14b/release/deepscaler/data/train.parquet \
  --destination /mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/assets/native-model \
  --manifest manifests/model.json
```

Preparation never overwrites an existing model view. The package includes `assets/train-manifest-v3.json` when built after local data validation. If that manifest is missing, prepare it using the configured native tokenizer:

```bash
vendor/prime-rl/.venv/bin/deepseek-study prepare-data configs/run.json
```

A changed prompt, token limit, grader source or grader dependency version requires a newly prepared manifest at a new path. The command refuses to replace an existing manifest.

The original cleaned dataset is pinned by revision and SHA256 and remains unchanged. The v3 preparation accepts **37,696 of 37,713 questions** and records 17 exclusions, including seven newly excluded structured time/score references. No labels are rewritten. The v2 reports in `diagnostics/prepared-data-*.json` describe the previous 37,703-question selection; the new selection and delta are recorded in [hardening-20260927.json](diagnostics/hardening-20260927.json). A changed selection changes prompt indexing and must be treated as a new study protocol. Preparation does not establish benchmark decontamination.

The local grader uses the last complete boxed answer after the closing thinking tag for native DeepSeek/Qwen3 reasoning. For base Qwen2.5-3B, `reasoning_required=false` permits a plain final box; any generated reasoning section must still close. This explicit model-format setting is bound into the prepared manifest and run configuration. A complete correct final answer may earn reward on a truncated response. Persistent isolated workers have an eight-second internal timeout and ten-second outer deadline, with one retry on the identical saved response. Infrastructure failures stop the cohort rather than becoming zero rewards. The teammate's v3.2 grader is not required or used.

The current reward policy is `strict-final-box-v3-context-clock-native-reasoning`. It preserves the structure/case safeguards and adds contextual clock comparison, finite-reference checks and self-verification during preparation. See [grading policy v3](docs/grading-v3.md) for exact boundaries, the limitation of references without AM/PM, and migration requirements. The stored 14B and 1.7B failed responses now grade correctly; a saved base-3B cohort has 76/512 correct responses and 30/64 groups with mixed rewards under the corrected format policy. This replay is not a new training run or an estimate of benchmark accuracy.

## Resolve, check and launch

```bash
vendor/prime-rl/.venv/bin/deepseek-study build configs/run.json resolved-configs
vendor/prime-rl/.venv/bin/deepseek-study check configs/run.json
vendor/prime-rl/.venv/bin/deepseek-study run configs/run.json
```

`build` resolves real official configurations without starting training. `check` verifies original asset identity, native tokenizer parity and the prepared data contract. `run` requires a new output directory, a Linux GPU runtime and exactly the configured visible GPUs. Execute inside the matching Slurm allocation. The launcher starts the official inference and environment servers, the controller, and the official trainer through a small RNG-checkpoint adapter. It terminates its registered process groups on failure or interruption with bounded shutdown waits. Structured final status retains the original error, signal, Slurm job identity and child exit codes; late observer cleanup warnings remain in the launcher log.

Each run copies the authored source into `output/source` and executes children from that snapshot. The run identity binds source files, the official lockfile/commit, runtime versions and the prepared data hash. The original official checkout remains unchanged.

Controller deadlines are separate: 30 minutes for packing/batch dispatch, up to four hours waiting for the next learner publication, 30 minutes for weight transfer, and two hours for checkpoint completion. The upstream publication/collective timeout also covers a legitimate generation wait and is therefore at least the 24-hour generation budget. `study/training_wait_seconds` measures the remaining wait after cohort generation, not total GPU training time.

## Recovery and evidence

Checkpoints occur every 100 completed optimizer updates and at completion, including an off-interval final update. The update count includes bootstrap. Every 100-step checkpoint is retained; retention also keeps the last four completed checkpoints, including the final checkpoint. A default 1,000-update run therefore keeps steps 100, 200, ..., 1,000.

With the default cluster root, checkpoints are written directly to NFS at `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/outputs/<run-name>/checkpoints/step_<N>/`. NFS is shared disk storage. Changing `output_dir` or the `init --root` argument changes this destination. Saving is synchronous and can pause progress while NFS writes finish; it does not add optimizer updates or change rollout age.

A checkpoint becomes complete after trainer state, every trainer rank's RNG state, sampler progress and the pending queue have been saved. Trainer state includes sharded model weights, optimizer state and scheduler state. Component manifests bind metadata/sampler/RNG contents and trainer shard sizes. They do not checksum every large tensor shard. These are distributed recovery checkpoints, not standalone Hugging Face model exports.

Intermediate evaluation is explicitly disabled. The saved milestones are available for a separate evaluation workflow after training; no evaluation job is automatically launched. Existing run JSON files retain their old interval: set `checkpoint_interval` and `checkpoint_keep_interval` to `100` and rebuild resolved configs before a new run, or use `init` for a new configuration. The current cluster release includes these defaults.

For resume, copy the same scientific configuration and change only `output_dir` to a new directory:

```bash
vendor/prime-rl/.venv/bin/deepseek-study run configs/resume.json --resume /absolute/path/to/old-run/checkpoints/step_100
vendor/prime-rl/.venv/bin/deepseek-study audit /absolute/path/to/new-run
```

Changed source, runtime identity, data or configuration is rejected. The original GPU layout is required; cross-layout optimizer/RNG resharding is not enabled. Queued responses are preserved exactly and trainer RNG is restored. Future inference is not guaranteed bitwise identical after restart because the complete vLLM RNG state is not restored. GPU backward is not configured for strict determinism either: the readiness replay matched all saved forward log probabilities but showed small post-update weight/Adam differences, documented in the [readiness report](docs/readiness-test.md). Only load trusted project checkpoints.

`updates.jsonl` records versions, exact age, bootstrap status, original question/response IDs, reward, zero-advantage fraction and queue accounting. `generations.jsonl` records generation time and provenance. `grading.jsonl` records extraction/comparison reasons and retry information. `rollouts/*.msgpack` preserves full training token/log-probability payloads; failed episodes are written under `failures/`. Original trainer metrics and process logs remain local. The TensorBoard observer records numeric metrics locally and mirrors event files to XFS; it does not upload rollout text or checkpoint files.

Rollout archive format 2 explicitly stores `sample_response_ids` and `sample_task_keys` in training-sample order. Other cohort metadata remains in arrival order and must be joined by response ID. Recovery preflight releases its temporary queue before workers start, and queue checkpoint I/O streams pickle data instead of creating a full serialized byte copy. The pending queue itself remains in CPU memory.

The benchmark suite from the teammate note is not bundled or automatically run: its frozen benchmark artifacts and audited exclusions were not supplied. Training metrics are not benchmark evaluation results.

## TensorBoard

The launcher starts a separate TensorBoard observer reading existing journals. It logs trainer metrics, consumed-rollout rewards, exact staleness, queue size, generation timing, inference telemetry, paper diagnostics, noncontributing-token fractions and checkpoint milestones. Runboard is disabled, including when its environment variables are inherited. Evaluation remains offline only.

TensorBoard 2.20.0 is pinned in this package. Events are stored in `<output_dir>/tensorboard/` on NFS and mirrored under the run's XFS metric directory. The observer does not change the training algorithm. See the [TensorBoard guide](docs/tensorboard.md) for viewing, metric clocks, replay and shutdown behavior.

## Local validation and portable package

```bash
uv run --no-project vendor/prime-rl/.venv/bin/python -m pytest -q tests --disable-warnings
uv run --no-project vendor/prime-rl/.venv/bin/python -m ruff check src tests scripts
uv run --no-project --python 3.12 scripts/package.py
```

The archive under `dist/` contains source, tests, configuration/schema, manifests, reports and the prepared question manifest. It excludes model weights, raw data, virtual environments, credentials and previous runs. Bootstrap fetches the official pinned dependency sources on the destination machine. `PACKAGE_SHA256.json` records the packaged file hashes.

See the [review guide](docs/review-guide.md), [architecture](docs/architecture.md), [cluster operations](docs/cluster.md) and [production continuation](docs/production-continuation.md).

## BAPO and M2PO research measurements

See the [complete paper metric inventory](docs/paper-metrics.md) for each figure/table, exact formulas, dashboard series and limits. Every update captures detached token ID, position, current/behavior log probabilities, advantage, full-vocabulary entropy and direct GRPO contribution masks in compressed per-rank files. A CPU observer calculates global diagnostics and feeds TensorBoard. It also mirrors metric evidence to `/mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics/<run-name>/`; checkpoints stay on NFS. The setting is `metrics_mirror_root`.

PrimeRL source remains unchanged. A second in-memory factory override, `setup_token_exporter`, activates compressed capture from its existing hooks. TensorBoard receives each bin as a scalar series; raw token arrays and frequency artifacts remain available for scientific plotting. These changes and their costs are documented in the [dependency guide](docs/upstream-integration.md). No training or intermediate evaluation is started by this integration.

Use `deepseek-study paper-metrics RUN_DIRECTORY --once` to replay completed raw evidence. Use `deepseek-study import-evaluation RUN_DIRECTORY predictions.jsonl protocol.json --step 100` for independently produced benchmark predictions. The importer validates coverage and provenance and logs benchmark accuracy; it is not a checkpoint exporter or evaluation inference runner.

The primary inactive-token metric is `paper/gradient_signal/noncontributing_token_fraction`: zero direct GRPO coefficients divided by valid response tokens. It includes zero advantages and effective clipping, while keeping the outside-bound fraction separate. Loss values, gradients and the exact-k scheduler are unchanged by the logging addition. [Metric definition](docs/paper-metrics.md#tokens-with-no-direct-grpo-contribution).

See the [three smaller-model full runs](docs/small-model-full-runs.md) for their pinned models, layouts, full budgets, launch procedure and recovery limits.
