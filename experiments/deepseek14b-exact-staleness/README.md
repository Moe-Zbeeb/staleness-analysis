# DeepSeek 14B / cleaned DeepScaleR / exact staleness

One run at a time, with the exact nonnegative integer `k` you request. This package composes official PrimeRL v0.9.0 at `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1`; it does not import the teammate fork or edit upstream files.

**Status:** the organized package passes all 103 tests locally. The earlier deployed layout passed the same suite on Linux, plus checks on eight A100 80GB GPUs for NCCL, BF16 backward, Flash Attention backward and vLLM RMSNorm. Those cluster receipts describe the earlier deployment, not a new test of this layout. Full 14B training, live model-weight transfer, memory fit and training/resume execution remain unverified. No study training has been launched. See [validation evidence](diagnostics/cluster-validation.json).

## Review and organization

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
| Taskset plugin | [deepseek_deepscaler/](src/deepseek_deepscaler/) | Connecting prepared questions and rewards to Verifiers |
| Installation and health checks | [scripts/](scripts/) | Preparing dependencies, packaging or checking hardware |
| Tests and evidence | [tests/](tests/), [diagnostics/](diagnostics/) | Reviewing validated behavior and limits |

Configuration templates/schema live in `configs/`; pinned model hashes are in `manifests/`. Preparation writes the question-selection record to `assets/train-manifest.json`. Weights, raw datasets, generated manifests, environments, caches, run outputs and job logs are excluded from Git.

The [staleness research audit](docs/staleness-research-audit.md) separates exact age, policy difference, learning quality and throughput. It maps the relevant knobs, differences from the cited paper, current measurement gaps and a proposed research protocol. It does not launch experiments or change the baseline.

## Imported libraries: what we change

**No tracked source files in official PrimeRL or its imported submodules are modified.** We pin PrimeRL v0.9.0 at `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1` and install its frozen dependency lock. The launcher rejects tracked changes or a different dependency revision.

We do customize behavior through our package: a configured loss, a group-advantage algorithm, a finite rollout source and explicit weight synchronization. In each trainer process, `runtime/trainer.py` temporarily replaces PrimeRL's checkpoint-manager factory with our RNG-saving adapter and restores the factory on exit. This is an in-memory override, not a source patch. A separate prepared model view changes only tokenizer-class metadata for compatibility; the original model files remain untouched.

The [integration guide](docs/upstream-integration.md) lists every customization, the dependency pins, and what must be reviewed when upgrading the library. These internal interfaces are version-sensitive; the package does not automatically follow upstream changes.

## Source-layout update

Implementation files now live under four concern-specific subpackages. The `deepseek-study` command, configuration fields, taskset name, scientific defaults and training algorithm are unchanged. Both hardware profiles resolve identically to the previous layout except for the custom loss import path, now `deepseek_study.learning.loss.clipped_grpo`. Advantage, loss, queue and reward implementation files retain their previous bytes; the grader identity and prepared dataset manifest are unchanged.

Regenerate resolved configs from the study JSON using `deepseek-study build`; do not reuse old resolved files with the flat loss import path. Source fingerprints and pickled module paths changed, so old source snapshots/checkpoints require their original code. Use a fresh source deployment for this layout; do not overlay it onto an old source tree and leave obsolete modules behind. [Upgrade and deployment notes](docs/upstream-integration.md#upgrades-and-layout-migration).

## Training contract

`theta_t` is the model after `t` optimizer updates. After the initial bootstrap, training `theta_t` consumes responses generated entirely by `theta_(t-k)`. Microbatches accumulate into one update; responses are trained on once. For `k=32`, updating `theta_100` uses responses from `theta_68`.

A fresh run first performs `k` explicitly labeled on-policy bootstrap updates while generating distinct future cohorts. Update `k+1` is the first exact-k update. With `k=0`, every update is on-policy. The total budget includes bootstrap: a 1,000-update run at k=32 has 32 bootstrap and 968 exact-k updates. A budget that leaves no exact-k updates is rejected.

Generation uses frozen current inference weights and queues its original responses until the prescribed update. Training and generation overlap where possible. Inference weights change only after the complete cohort finishes. Tail generation stops early enough that no extra cohorts remain after the final update. Cohort prompts are indexed by their intended consumption update, so changing k does not change the assigned prompt stream.

## Baseline recipe

The `init` command writes a complete configuration. Its defaults are 64 prompts × 8 responses = 512 responses per update, a 2,048-token rendered-prompt cap, 8,192 response tokens, seed 42, and 1,000 total updates. Sampling uses temperature 1, top-p 1, disabled top-k/min-p, and neutral penalties.

The loss is clipped GRPO with epsilon 0.2, reward-minus-group-mean advantages without standard-deviation normalization, and a global token mean. Reference KL and entropy coefficients are zero. Zero-advantage groups remain in the denominator. An all-zero cohort still executes AdamW; existing optimizer momentum may move the policy even with weight decay disabled.

AdamW uses LR 1e-6, betas 0.9/0.999, epsilon 1e-8, weight decay 0 and gradient norm clipping at 1. Disabling weight decay removes parameter shrinkage independent of the GRPO loss gradient; the exact-age queue and GRPO objective are unchanged. LR warms up over 30 optimizer updates, then remains constant. This LR warm-up is distinct from the k-update bootstrap. Full-model training uses BF16 compute, FP32 optimization/reduction, activation checkpointing, and no quantization. Compilation and prefix caching are initially disabled.

The baseline and configuration template previously used weight decay 0.01. Existing run JSON files retain their explicit settings: set `weight_decay` to `0.0` and rebuild resolved configs before a new run, or create a new configuration with `init`. This is a scientific configuration change, so it must not be applied while resuming an older recipe. The cluster deployment and its historical validation receipts predate this change.

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

Preparation never overwrites an existing model view. The package includes `assets/train-manifest.json` when built after local data validation. If that manifest is missing, prepare it using the configured native tokenizer:

```bash
vendor/prime-rl/.venv/bin/deepseek-study prepare-data configs/run.json
```

A changed prompt, token limit, grader source or grader dependency version requires a newly prepared manifest at a new path. The command refuses to replace an existing manifest.

The original cleaned dataset is pinned by revision and SHA256 and remains unchanged. Deterministic preparation accepts **37,703 of 37,713 questions**. Ten unsupported references are excluded with original IDs, reference text and reasons; no labels are rewritten. No questions exceeded the 2,048-token prompt cap; the longest rendered prompt was 834 tokens. See `diagnostics/prepared-data-exclusions.json` and `diagnostics/prepared-data-summary.json`. This validation does not itself establish benchmark decontamination.

The local grader uses the last complete boxed answer after the closing thinking tag, strips only outer math delimiters from references, disables permissive parse fallback, and gives binary correctness reward. A complete correct final answer may earn reward on a truncated response. Persistent isolated workers have an eight-second internal timeout and ten-second outer deadline, with one retry on the identical saved response. Infrastructure failures stop the cohort rather than becoming zero rewards. The teammate's v3.2 grader is not required or used.

## Resolve, check and launch

```bash
vendor/prime-rl/.venv/bin/deepseek-study build configs/run.json resolved-configs
vendor/prime-rl/.venv/bin/deepseek-study check configs/run.json
vendor/prime-rl/.venv/bin/deepseek-study run configs/run.json
```

`build` resolves real official configurations without starting training. `check` verifies original asset identity, native tokenizer parity and the prepared data contract. `run` requires a new output directory, a Linux GPU runtime and exactly the configured visible GPUs. Execute inside the matching Slurm allocation. The launcher starts the official inference and environment servers, the controller, and the official trainer through a small RNG-checkpoint adapter. It terminates its process groups on failure or interruption.

Each run copies the authored source into `output/source` and executes children from that snapshot. The run identity binds source files, the official lockfile/commit, runtime versions and the prepared data hash. The original official checkout remains unchanged.

## Recovery and evidence

Checkpoints occur every 100 completed optimizer updates and at completion, including an off-interval final update. The update count includes bootstrap. Every 100-step checkpoint is retained; retention also keeps the last four completed checkpoints, including the final checkpoint. A default 1,000-update run therefore keeps steps 100, 200, ..., 1,000.

With the default cluster root, checkpoints are written directly to NFS at `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/outputs/<run-name>/checkpoints/step_<N>/`. NFS is shared disk storage. Changing `output_dir` or the `init --root` argument changes this destination. Saving is synchronous and can pause progress while NFS writes finish; it does not add optimizer updates or change rollout age.

A checkpoint becomes complete after trainer state, every trainer rank's RNG state, sampler progress and the pending queue have been saved. Trainer state includes sharded model weights, optimizer state and scheduler state. Component manifests bind metadata/sampler/RNG contents and trainer shard sizes. They do not checksum every large tensor shard. These are distributed recovery checkpoints, not standalone Hugging Face model exports.

Intermediate evaluation is explicitly disabled. The saved milestones are available for a separate evaluation workflow after training; no evaluation job is automatically launched. Existing run JSON files retain their old interval: set `checkpoint_interval` and `checkpoint_keep_interval` to `100` and rebuild resolved configs before a new run, or use `init` for a new configuration. The earlier cluster deployment has not received these changes.

For resume, copy the same scientific configuration and change only `output_dir` to a new directory:

```bash
vendor/prime-rl/.venv/bin/deepseek-study run configs/resume.json --resume /absolute/path/to/old-run/checkpoints/step_100
vendor/prime-rl/.venv/bin/deepseek-study audit /absolute/path/to/new-run
```

Changed source, runtime identity, data or configuration is rejected. The original GPU layout is required; cross-layout optimizer/RNG resharding is not enabled. Queued responses are preserved exactly and trainer RNG is restored. Future inference is not guaranteed bitwise identical after restart because the complete vLLM RNG state is not restored. Only load trusted project checkpoints.

`updates.jsonl` records versions, exact age, bootstrap status, original question/response IDs, reward, zero-advantage fraction and queue accounting. `generations.jsonl` records generation time and provenance. `grading.jsonl` records extraction/comparison reasons and retry information. `rollouts/*.msgpack` preserves full training token/log-probability payloads; failed episodes are written under `failures/`. Official trainer metrics and process logs remain local. No external tracking is enabled.

The benchmark suite from the teammate note is not bundled or automatically run: its frozen benchmark artifacts and audited exclusions were not supplied. Training metrics are not benchmark evaluation results.

## Local validation and portable package

```bash
uv run --no-project vendor/prime-rl/.venv/bin/python -m pytest -q tests --disable-warnings
uv run --no-project vendor/prime-rl/.venv/bin/python -m ruff check src tests scripts
uv run --no-project --python 3.12 scripts/package.py
```

The archive under `dist/` contains source, tests, configuration/schema, manifests, reports and the prepared question manifest. It excludes model weights, raw data, virtual environments, credentials and previous runs. Bootstrap fetches the official pinned dependency sources on the destination machine. `PACKAGE_SHA256.json` records the packaged file hashes.

See the [review guide](docs/review-guide.md), [architecture](docs/architecture.md) and [cluster operations](docs/cluster.md).
