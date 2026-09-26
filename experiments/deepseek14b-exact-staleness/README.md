# DeepSeek 14B / cleaned DeepScaleR / exact staleness

One run at a time, with the exact nonnegative integer `k` you request. This package composes official PrimeRL v0.9.0 at `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1`; it does not import the teammate fork or edit upstream files.

**Status:** deployed on the cluster; all 103 tests passed locally and on Linux. Runtime checks passed on eight A100 80GB GPUs for NCCL, BF16 backward, Flash Attention backward and vLLM RMSNorm. Full 14B training, live model-weight transfer, memory fit and training/resume execution remain unverified. No study training has been launched. See [validation evidence](diagnostics/cluster-validation.json).

## Review and organization

Start with the [review guide](docs/review-guide.md) for the module map and a worked exact-k example. Runtime source lives in `src/`, installation and health tools in `scripts/`, focused checks in `tests/`, and configuration contracts in `configs/`. Pinned model hashes are in `manifests/`; deterministic preparation writes the question-selection record to `assets/train-manifest.json`. Model weights, raw datasets, generated manifests, environments, caches, run outputs and job logs are excluded from Git. The small preparation summary and exclusion report remain reviewable in `diagnostics/`.

The [staleness research audit](docs/staleness-research-audit.md) separates exact age, policy difference, learning quality and throughput. It maps the relevant knobs, differences from the cited paper, current measurement gaps and a proposed research protocol. It does not launch experiments or change the baseline.

## Training contract

`theta_t` is the model after `t` optimizer updates. After the initial bootstrap, training `theta_t` consumes responses generated entirely by `theta_(t-k)`. Microbatches accumulate into one update; responses are trained on once. For `k=32`, updating `theta_100` uses responses from `theta_68`.

A fresh run first performs `k` explicitly labeled on-policy bootstrap updates while generating distinct future cohorts. Update `k+1` is the first exact-k update. With `k=0`, every update is on-policy. The total budget includes bootstrap: a 1,000-update run at k=32 has 32 bootstrap and 968 exact-k updates. A budget that leaves no exact-k updates is rejected.

Generation uses frozen current inference weights and queues its original responses until the prescribed update. Training and generation overlap where possible. Inference weights change only after the complete cohort finishes. Tail generation stops early enough that no extra cohorts remain after the final update. Cohort prompts are indexed by their intended consumption update, so changing k does not change the assigned prompt stream.

## Baseline recipe

The `init` command writes a complete configuration. Its defaults are 64 prompts × 8 responses = 512 responses per update, a 2,048-token rendered-prompt cap, 8,192 response tokens, seed 42, and 1,000 total updates. Sampling uses temperature 1, top-p 1, disabled top-k/min-p, and neutral penalties.

The loss is clipped GRPO with epsilon 0.2, reward-minus-group-mean advantages without standard-deviation normalization, and a global token mean. Reference KL and entropy coefficients are zero. Zero-advantage groups remain in the denominator. An all-zero cohort still executes AdamW; momentum and weight decay may move the policy.

AdamW uses LR 1e-6, betas 0.9/0.999, epsilon 1e-8, weight decay 0.01 and gradient norm clipping at 1. LR warms up over 30 optimizer updates, then remains constant. This LR warm-up is distinct from the k-update bootstrap. Full-model training uses BF16 compute, FP32 optimization/reduction, activation checkpointing, and no quantization. Compilation and prefix caching are initially disabled.

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

Checkpoints occur every 25 updates and at completion. A checkpoint becomes complete after trainer state, every trainer rank's RNG state, sampler progress and the pending queue have been saved. Component manifests bind metadata/sampler/RNG contents and trainer shard sizes. They do not checksum every large tensor shard. Retention removes only completed checkpoints from this run, keeping the last four and every 100th update.

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
