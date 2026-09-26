# Independent checkpoint evaluation

This evaluator runs beside the exact-staleness experiments, on separate A100 40GB GPUs. Each researcher owns one persistent pool, registers all of their training runs, and scales the pool's worker limit as needed. Training does not wait for evaluation.

**Validation status, September 26, 2026:** CPU checkpoint reconstruction, recovery tests and real 14B inference on an A100 40GB passed. A complete export-and-evaluate test of a teammate's trained 14B checkpoint is still pending read access to its completion/component metadata. The real checkpoint's tensor schema was checked successfully. No production training run was registered during development. See [validation and remaining acceptance](#validation-and-remaining-acceptance).

## What happens

```text
Training run A ── completed checkpoints ─┐
Training run B ── completed checkpoints ─┼─> CPU coordinator / durable queue
Training run C ── completed checkpoints ─┘        │
                                                v
                                      CPU model-only HF export
                                                │
                                                v
                                      A100 40GB evaluation workers
                                      node-local model cache
                                                │
                                                v
                                      raw responses + grades on NFS
                                      complete-checkpoint metrics
```

The evaluated model is the **current learner after the checkpoint's completed optimizer update**, `theta_t`. The evaluator reads the official PrimeRL distributed checkpoint's `app.model` tensors. It does not replay training responses, load historical actors, or allocate optimizer state. It never deserializes the historical rollout queue.

Discovery requires the training pipeline's format-2 `study/complete.json`, matching run/source identities and a verified component manifest. Partially written checkpoints are ignored. Source manifests hash metadata/RNG/sampler files and record large trainer-shard sizes; they do **not** checksum every source tensor byte. Exports add finite-value checks and SHA-256 checksums for every exported file.

The default training recipe saves and retains steps **100, 200, …, 1,000**, including bootstrap updates. The pool evaluates every discovered interval milestone. Registration checks retention compatibility; discovery currently follows multiples of `checkpoint_interval`, not arbitrary off-interval snapshots. Use the default 100/100 retention and a 1,000-update horizon for this experiment. The initial model is an explicitly scheduled, shared step-0 baseline.

The coordinator polls every 60 seconds. Tasks rotate fairly among registered runs, with older steps first within a run. Export must finish before its GPU tasks can start. One checkpoint becomes 36 question shards; all repetitions of a question stay together. Workers claim tasks from the pool rather than being permanently assigned to a lag. A worker starts a fresh inference process per shard; model files are reused through the local cache.

## Frozen evaluation protocol

| Setting | Value |
| --- | --- |
| Initialization | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`, revision `1df8507178afcc1bef68cd8c393f61a886323761` |
| Generation cap | **8,192 response tokens** |
| Rendered prompt cap | **2,048 tokens** |
| Total context | **10,240 tokens**; “10K context” does not mean 10K generated tokens |
| Prompt/template | Native training tokenizer/chat template, open reasoning channel, identical answer instruction |
| Sampling | Temperature 1, top-p 1, top-k disabled, min-p 0; neutral repetition/presence/frequency penalties |
| Seeds | Base seed 42; fixed seed per question and repetition, independent of lag/checkpoint/worker |
| Score | Mean per-question sampled pass@1; multiple samples estimate accuracy, not best-of-G |
| Grader | Corrected benchmark `math-sweep-final-v3.2`, with frozen corrections, exclusions and format hints |
| Training reward | Unchanged: the training pipeline uses its own strict-final-box grader |
| Backend | Pinned PrimeRL v0.9.0, PyTorch `2.11.0+cu128`, Transformers `5.6.2`, vLLM `0.26.0+cu129` |
| Inference | BF16 weights with PrimeRL's FP32 output-head extension, eager execution, no quantization or prefix caching |
| Worker | One A100 40GB, TP=1, at most 2 active sequences, GPU memory fraction 0.9 |

| Benchmark | Questions | Samples/question | Responses/checkpoint | Shards |
| --- | ---: | ---: | ---: | ---: |
| MATH500 | 500 | 4 | 2,000 | 32 |
| AIME 2024, audited subset | 29 | 16 | 464 | 2 |
| AIME 2025 | 30 | 16 | 480 | 2 |
| **Total** | **559** | — | **2,944** | **36** |

AIME24 excludes `aime24:88`, whose source wording incorrectly duplicates the tangency condition. Report it as **29-question audited AIME24**, not the unmodified 30-question benchmark. The source-bound [grading policy](../configs/evaluation-grading-policy.json) also contains corrections for benchmarks outside this core suite; those benchmarks are not evaluated by default. Counts and repetition schedules come from the shared frozen sweep manifest.

The protocol embeds the actual questions, references, revisions, row hashes, grader/dependency identities, evaluator source hashes, key upstream inference hashes and backend versions. Use the **same protocol file and identity** across runs and owners. Do not regenerate it independently from a newer dataset revision or silently change a token budget after an OOM.

Matching seeds do not guarantee token-identical repeated generation with the current multiprocessing backend. Two six-question GPU smoke runs produced identical tokens on four questions and different tokens on two; all six grades matched. This is a small functional check, not evidence of statistical equivalence. Retain raw outputs as the evidence of each run. See the pinned [vLLM reproducibility guidance](https://docs.vllm.ai/en/v0.26.0/usage/reproducibility/).

## Deploy without changing a running trainer

Use a **separate checkout and evaluator output directory**. Training resume validates its original source snapshot; adding evaluator files to a running trainer's source tree changes that identity. Keep existing training releases and runtimes intact.

On the cluster, clone the evaluator branch and record the exact commit before launching:

```bash
git clone --branch eval/checkpoint-pool --single-branch \
  https://github.com/Moe-Zbeeb/staleness-analysis.git staleness-checkpoint-evaluator
cd staleness-checkpoint-evaluator
git rev-parse HEAD
git checkout --detach HEAD
cd experiments/deepseek14b-exact-staleness
```

Bootstrap a separate Linux runtime using the official pinned lock. Run installation/build work and CPU tests in an appropriate allocation, not on a login node. Do not copy a macOS virtual environment to Linux.

```bash
uv run --no-project --python 3.12 scripts/bootstrap.py --gpu --attention fa2
```

Alternatively, point the evaluator at an existing **matching, frozen** Linux runtime. Do not upgrade an environment used by an active trainer. Runtime/source guards reject mismatches. The executable paths must be accessible on every worker node; `ninja`, the CUDA toolchain and `findmnt` must be available there.

Source your own cluster environment helper, if required, **before** setting the absolute paths below. Some helpers change the working directory. Run this block from the evaluator experiment directory, replacing the account path:

```bash
export EVAL_SOURCE="$(pwd)/src"
export EVAL_PYTHON="$(pwd)/vendor/prime-rl/.venv/bin/python"
export EVAL_UV="$(command -v uv)"
export PYTHONPATH="$EVAL_SOURCE"
export EVAL_ROOT="/mnt/nfs/home/YOUR_ACCOUNT/evaluations/core-8k-v1"
export EVAL_SLURM_CONFIG="$EVAL_ROOT/slurm.json"
# Optional: export EVAL_CLUSTER_ENV=/absolute/path/to/your/cluster-env.sh

eval_cli() {
  "$EVAL_UV" run --no-project "$EVAL_PYTHON" -m deepseek_study.evaluation "$@"
}

eval_cli init "$EVAL_ROOT" --tp 1 --max-sequences 2
mkdir -p "$EVAL_ROOT/slurm"
cp configs/evaluation-slurm.template.json "$EVAL_SLURM_CONFIG"
```

Edit the copied `slurm.json`: set absolute `python` and `uv` paths to the values above, and verify your account, partition, QoS and eligible nodes. The template uses `grad-students`, `low-priority`, `background`, and `deep-chungus-[1-8,10]`, the A100 40GB nodes checked for this deployment. Recheck the hardware inventory when using another cluster or after hardware changes.

The node list is an **eligibility set**, not a request for every listed node. The scheduler computes exclusions and asks for one eligible node per worker. Busy GPUs are fine: jobs stay pending until Slurm allocates them. Runtime checks also reject an unexpected GPU model, memory size or visible-device count.

### Prepare or obtain the shared protocol

A Git clone contains code and the correction policy, **not** model weights, checkpoint files, the original sweep manifest or a generated benchmark protocol. The initial handoff therefore needs these external assets:

1. A prepared native-tokenizer model directory matching training.
2. Either the team's frozen `core-8k.json`, or the original sweep manifest matching the bundled grading policy.
3. Read/traverse access to registered run metadata, model assets and checkpoint components.

With the original manifest, freeze once inside a CPU allocation with the pinned GPU runtime installed:

```bash
export EVAL_MODEL=/absolute/path/to/prepared-native-model
export EVAL_MANIFEST=/absolute/path/to/original-sweep/manifest.json
export EVAL_PROTOCOL="$EVAL_ROOT/core-8k.json"

eval_cli freeze-sweep "$EVAL_MANIFEST" configs/evaluation-grading-policy.json \
  "$EVAL_MODEL" "$EVAL_PROTOCOL"
```

`freeze-sweep` refuses an existing destination and rejects manifest/policy mismatches. Alternatively copy the **already frozen** protocol byte-for-byte to `EVAL_PROTOCOL`; it contains the question set and does not need the original sweep manifest at worker runtime. Preserve the same evaluator source and runtime. Share only the required assets/metadata with the relevant accounts; each account can also run its own pool against its own checkpoints.

Schedule the initial model once per pool/protocol. This copies about 30 GB and belongs in a CPU allocation:

```bash
eval_cli baseline "$EVAL_ROOT" "$EVAL_MODEL" "$EVAL_PROTOCOL"
```

### Register training runs

Register each **training output directory**, after its `configs/study.json` and `source/identity.json` exist. The training configuration supplies lag, seed, model path, update horizon and checkpoint cadence. Use absolute original paths, not a relocated output copy whose stored `output_dir` differs.

```bash
eval_cli register "$EVAL_ROOT" /absolute/path/to/outputs/lag0-seed42 "$EVAL_PROTOCOL"
eval_cli register "$EVAL_ROOT" /absolute/path/to/outputs/lag256-seed42 "$EVAL_PROTOCOL"
eval_cli discover "$EVAL_ROOT"
```

Registration and discovery are idempotent. Add more runs with `register` while the coordinator is running. If the run belongs to a teammate, the required private completion/component files and referenced assets must be readable; readable trainer shards alone are insufficient. Failed discovery is recorded in `status.json`, not converted into benchmark results. Register a resumed run's new output directory separately; reports retain distinct run identities and do not automatically stitch runs together.

### Launch the pool

Run the following from the experiment directory after exporting the variables above. Submission itself can happen on the login node:

```bash
sbatch --parsable --account=grad-students --partition=low-priority --qos=background \
  --export=ALL --output="$EVAL_ROOT/slurm/coordinator-%j.log" \
  scripts/evaluation_coordinator.sbatch
```

Record the returned coordinator job ID. The coordinator submits its own finite worker jobs. The laptop/SSH session can disconnect after Slurm accepts the submission.

| Component | Resources per job | Default simultaneous jobs | Lifetime |
| --- | --- | ---: | --- |
| Coordinator | 1 CPU, 2 GB host RAM, no GPU | 1 per pool | 14 days; requests requeue before walltime |
| Exporter | 8 CPUs, 32 GB host RAM, no GPU | 1 | Up to 12 hours |
| Evaluator | 8 CPUs, 64 GB host RAM, **1 A100 40GB** | 2 | Up to 12 hours |

The two evaluator jobs consume two additional GPUs outside the training allocations. Increase `limits.evaluate` in `slurm.json` to allow more workers, then restart the coordinator; limits are read at startup. Already submitted workers continue. Scheduler pending jobs count toward the limit. Start small and measure checkpoint backlog and NFS traffic before increasing concurrency. TP=2 is exposed by the CLI but has **not** received the TP=1 GPU acceptance test; changing the worker profile requires a new pool and validation.

## Monitor, recover and stop

```bash
eval_cli status "$EVAL_ROOT"
eval_cli collect "$EVAL_ROOT"
# Combine completed results from readable pools; run this periodically, not in a GPU worker.
eval_cli collect /absolute/path/to/pool-A /absolute/path/to/pool-B
```

The coordinator writes `status.json` and `metrics.json` each polling cycle. Monitor coordinator liveness, discovery errors, scheduler errors, failed tasks and oldest pending checkpoint together. A live trainer does not imply a live evaluator.

| Location under the pool | Contents |
| --- | --- |
| `runs/`, `protocols/`, `profile.json` | Frozen run registrations, scientific protocol and worker profile |
| `queue.json` | Task states, attempts, dependencies and errors |
| `submissions/<job>.json`, `slurm/*.log` | Submitted commands, job IDs and Slurm output |
| `attempts/<task>/<lease>/worker.log` | Child-process diagnostics for each attempt |
| `exports/<checkpoint-id>/` | Immutable HF model export and `evaluation-ready.json` |
| `results/<task>/raw/`, `grade/` | Checksummed responses/token IDs and grading records |
| `results/<task>/complete.json` | Durable shard-completion receipt |
| `metrics.json` | Complete-checkpoint benchmark rows and pending task information |

Workers heartbeat about every 20 seconds; leases expire after 180 seconds. If an attempt makes no durable progress for 30 minutes, its process group is terminated and the task retries. Each task gets three attempts before remaining failed. Expired owners cannot publish results after another worker takes over. Saved responses and grades are reused, so a grader failure does not require regenerating an already saved response. Work that died before a durable save may be generated again.

Fix the cause before explicitly retrying an exhausted task:

```bash
eval_cli retry "$EVAL_ROOT" TASK_ID_FROM_STATUS
```

Failed exports block their evaluation tasks; retry the export first. Persistent OOMs, disk exhaustion, permissions or source/runtime mismatches require intervention. Recovery never silently changes precision, token limits or the benchmark. The coordinator tolerates transient scheduler command failures, but an invalid configuration or an unhandled filesystem error can stop it; this is bounded recovery, not an external availability monitor.

To pause submissions, cancel **the recorded coordinator job ID**. Existing workers continue until they finish or reach their job limit. To stop workers too, inspect this pool's submission records and `squeue`, then cancel those specific job IDs. Restart with the same paths and launch command to resume. Do not cancel all of an account's jobs, since those may include training. Before changing worker limits, wait until the old coordinator has stopped so its singleton lock is released.

## Storage and throughput

Keep checkpoints, exports, the queue and response evidence on persistent shared storage. The worker stages models to an owned `/tmp/deepseek-evaluation-<uid>` directory and verifies its filesystem type. The checked CSAIL nodes had local ZFS under `/tmp`; both `/mnt/nfs` **and `/mnt/xfs` were NFS mounts**. A directory named XFS is not proof of node-local storage.

Staging verifies file hashes and requires enough free local disk for the incoming model plus 5 GiB. Cache cleanup targets two recent model directories while preserving any in use; active models can exceed that count. Two shared I/O slots per pool limit export/staging traffic. Different researchers' pools do not share that throttle, so coordinate worker limits when the cluster is busy.

Budget approximately **30 GB per BF16 checkpoint export**, or roughly **300 GB for ten checkpoints per run**, in addition to training recovery checkpoints, the baseline and raw results. The evaluator does not automatically delete NFS exports or training checkpoints. Keep source checkpoints retained while exports are pending. Cleanup of permanent evidence is an explicit later operation. Do not put `EVAL_ROOT` inside the source checkout or edit frozen source files while workers are running.

## Reading the results

`collect` publishes benchmark rows only when every shard, question and repetition for that checkpoint is complete. Partial results stay pending; a half-finished checkpoint is not a comparable benchmark score. Each row carries run identity, lag, training seed, completed optimizer step, exact-stale-update count, protocol/checkpoint identities and completion time.

| Metric | Meaning |
| --- | --- |
| `accuracy` | Average, across questions, of that question's correct-sample fraction; multiply by 100 for percent |
| `question_standard_error` | Standard error of those per-question fractions, not variation across independent training seeds |
| `mean_output_tokens` | Mean generated length including capped/truncated samples |
| `mean_finished_output_tokens` | Mean length only for samples that did not hit the generation cap |
| `truncated_count`, `truncated_rate` | Responses with backend finish reason `length` |
| `missing_final_count`, `missing_final_rate` | Responses without an extractable complete final box under the grader's extraction rule |
| `unsupported_count`, `unsupported_rate` | Responses classified as unsupported by the comparator |

Missing-final and truncation are separate, overlapping categories. A capped response may contain a complete correct final answer and receive credit. Infrastructure failures cannot become ordinary wrong answers: they prevent completion. Inspect unsupported cases separately from backend/grader failures.

Raw text, prompt/output token IDs, seeds, finish reasons, hardware/runtime provenance and grades are retained. They support later audits or regrading without regeneration; a general regrading command is not included in this release. There is no automatic Slack delivery or live Runboard benchmark upload. The existing `deepseek-study import-evaluation` has a different paper-suite schema and lifecycle; do not pass `collect` output to it directly.

## Validation and remaining acceptance

| Check | Recorded result |
| --- | --- |
| Local full project suite | **213 passed, 2 skipped**; Ruff and coordinator shell syntax passed |
| Linux evaluator suite | **45 passed**, CPU job `2144974` |
| Real scheduler CPU launch/export | Job `2144972` completed using a 27-tensor fixture |
| Real 14B worst-context inference | Job `2144970`: one A100 PCIe 40GB, two 2,048-token prompts and two forced 8,192-token outputs; 219.1 seconds for generation |
| Real benchmark smoke | Jobs `2144970` and `2144973`: six responses each, three shards each, no retries; completed-task restart generated zero extra responses |
| Production GPU submission path | Job `2144973` used the actual Slurm worker launcher |
| Final collector revision | Only `report.py` changed after the last GPU job; all 12 saved outputs verified, plus a 60-result memory regression |
| Teammate's real checkpoint schema | **579 tensors / 14,770,033,664 parameters**, FP32 storage, expected HF Qwen2 keys and shapes |
| Full trained-checkpoint export → inference | **Pending private metadata access**; schema compatibility alone does not certify this path |

The suite exercises the official PrimeRL `AppState` checkpoint format, actual two-rank Gloo/DTensor reconstruction, exact BF16 tensor and same-backend CPU-logit parity on fixtures, incomplete/corrupted checkpoint rejection, prompt parity with the official renderer, grader regressions, fair scheduling, lease fencing, stalled-process cleanup, cache verification, durable retries and complete-only aggregation. The two inherited local skips cover unavailable GPU/prepared-asset conditions. The 40GB evaluation result does not validate the separate seven-trainer-GPU 40GB **training** profile, nor establish full-suite throughput or an ETA.

Run the tests from the experiment directory in a CPU allocation:

```bash
uv run --no-project vendor/prime-rl/.venv/bin/python -m pytest tests/test_evaluation_pool.py -q
uv run --no-project vendor/prime-rl/.venv/bin/python -m pytest -q
uv run --no-project vendor/prime-rl/.venv/bin/ruff check src scripts tests
bash -n scripts/evaluation_coordinator.sbatch
```

For a new GPU deployment, prepare a **separate diagnostic pool** with two questions per benchmark and one sample each, retaining the full 8,192-token cap. `MODEL_RECORD` is the original sweep model record containing its revision and local model path:

```bash
export EVAL_SMOKE=/absolute/persistent/path/to/new-evaluation-smoke
export MODEL_RECORD=/absolute/path/to/sweep/model-record.json
"$EVAL_UV" run --no-project "$EVAL_PYTHON" scripts/evaluation_smoke.py prepare "$EVAL_SMOKE" \
  --model-record "$MODEL_RECORD" --manifest "$EVAL_MANIFEST" \
  --policy configs/evaluation-grading-policy.json
# Run this second command inside an allocation with exactly one visible A100 40GB:
"$EVAL_UV" run --no-project "$EVAL_PYTHON" scripts/evaluation_smoke.py gpu "$EVAL_SMOKE"
```

The GPU phase runs the full-context stress in its own process, then evaluates and grades all six smoke responses and verifies restart idempotence. Use at least 8 CPUs, 64 GB host RAM and enough local disk. Smoke scores are **not** benchmark estimates.

To close the remaining integration check, the checkpoint owner must grant read/traverse access to one completed checkpoint's `study/complete.json`, `study/components.json`, all referenced component files and training/model metadata, or run the check under their own account. In a diagnostic pool, register the run, discover it, and run `work ROOT --kind export --once` inside a CPU allocation. Then run `work ROOT --kind evaluate --once` on the validated GPU profile and verify the durable receipt. This proves the first real checkpoint export and one real evaluation shard; let the pool finish all shards before reporting its benchmark scores. `scripts/evaluation_checkpoint_probe.py TRAINER_DIRECTORY PREPARED_MODEL RECEIPT_JSON` checks names/shapes only and is not a substitute.

### Existing CSAIL handoff assets

These are dated owner-specific paths, not files bundled in Git. Access may need to be shared explicitly:

| Asset | Path |
| --- | --- |
| Final development release | `/mnt/nfs/home/bazzim/projects/checkpoint-evaluation-20260926/releases/dev04` |
| Acceptance receipt | `/mnt/nfs/home/bazzim/projects/checkpoint-evaluation-20260926/acceptance-result.json` |
| Frozen full core protocol | `/mnt/nfs/home/bazzim/projects/checkpoint-evaluation-20260926/smoke-v4/core-8k.json` |
| Verified model-only baseline | `/mnt/nfs/home/bazzim/projects/checkpoint-evaluation-20260926/smoke-v1/baseline` |
| Original sweep manifest | `/mnt/nfs/home/bazzim/projects/math-init-sweep-20260923/runs/75c1e4724dbd17e7/manifest.json` |
| Original model record | `/mnt/nfs/home/bazzim/projects/math-init-sweep-20260923/assets/models/deepseek-ai--DeepSeek-R1-Distill-Qwen-14B.json` |

The final core protocol identity is `c61cf85b92d890d4d0b07d2504045872e9a8f510530026fdb806685d2052e162`. Use that identity for this comparison. The `smoke-v4/pool` directory is a prepared diagnostic pool, not an activated production service. Create a distinct persistent pool for the actual experiments.
