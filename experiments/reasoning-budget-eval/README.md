# Five-model reasoning budget evaluation

The first training-pool evaluation: **256 random questions from each cleaned Skywork and DeepScaleR pool, five models, one response per question at each 4,096/8,192/12,288-token cap**. This produces 7,680 responses across 30 cells. The merged pool and the later 1,000-question, eight-rollout experiment are outside this directory.

Sampling is without replacement with seed 42. All models and budgets use the same selected questions. Each budget is a fresh decoding call with the same per-question seed; identical prefixes across calls are not guaranteed. Caps include reasoning and final-answer tokens. There is no minimum response length or prompt truncation. Qwen3 thinking is enabled. Native chat templates are retained, including Phi-4's built-in system prompt.

| Tag / model-directory name | Model | Temperature | Top-p | Top-k |
|---|---|---:|---:|---:|
| `deepseek` | DeepSeek-R1-Distill-Qwen-14B | 0.6 | 0.95 | disabled |
| `qwen3` | Qwen3-14B, thinking enabled | 0.6 | 0.95 | 20 |
| `acereason` | AceReason-Nemotron-14B | 0.6 | 0.95 | disabled |
| `openreasoning` | OpenReasoning-Nemotron-14B | 0.6 | 0.95 | disabled |
| `phi4` | Phi-4-reasoning-plus | 0.8 | 0.95 | 50 |

`experiment.json` records full model repository IDs, immutable revisions, file inventories, weight checksums and sampling settings. `datasets.lock.json` identifies the published cleaned Parquet releases, revisions, row counts and checksums. Model-specific settings are fixed across budgets; this is not a comparison at identical sampling settings.

## Grading and metrics

Math-Verify **0.8.0** compares the final answer after `</think>` against the stored reference, with strict mathematical parsing and no string fallback. Unfinished reasoning and unparseable predictions count as incorrect. Multiple required components are not treated as alternative acceptable answers. Accuracy always uses all 256 sampled questions in each cell.

One sampled Skywork reference is a symbolic maximum involving cyclic variation that the parser cannot safely extract. Such cases remain unresolved, with lower/upper accuracy bounds, until reviewed. Manual grades retain a rationale and must match the SHA256 of the exact completion; they cannot be reused against a new generation just because its question ID and budget match. The metric evaluates final-answer equivalence, not proof completeness. A response stopped by the cap can still be correct if its final-answer section already contains the correct answer.

Outputs include correct and pending counts, accuracy and Wilson 95% intervals, all-response mean/median/p90 generated lengths, total generated tokens, budget-hit rate, parse rate, and `finished_count` / `mean_finished_response_tokens`. The finished-only mean includes responses with `finish_reason=stop` and excludes cap truncations; it is conditional on finishing, not a population-wide length estimate. These are diagnostic scores on sampled training pools, not held-out benchmark results.

## Reproduce in a fresh directory

Use Python 3.12. Run substantial data/model verification, preparation and scoring in a CPU allocation with eight CPUs available; run generation in the GPU allocation below. Keep preparation/scoring separate from the existing vLLM runtime because their Transformers requirements differ. `runtime-versions.json` records the observed GPU versions; it is not a complete GPU wheel lock or a promise of bit-identical reproduction on different hardware.

1. Obtain the exact cleaned Parquet releases from `datasets.lock.json`, or use the sibling [dataset-preparation project](../exact-age-14b-data/README.md). Supply `DATASET_ROOT/data/processed/{skywork,deepscaler}/train.parquet`. The merged lock entry is retained as collection provenance but never sampled here.
2. Obtain the pinned model snapshots listed in `experiment.json`. Supply one directory or symlink per model tag beneath `MODELS_ROOT`, for example `MODELS_ROOT/deepseek/config.json`. Model weights are not downloaded by this evaluator.
3. Create isolated preparation and scoring environments. Point `GENERATION_PYTHON` at an existing compatible vLLM environment. No script installs or changes the GPU environment.

From this directory, configure and prepare:

```bash
python3.12 -m venv .venv-prep
.venv-prep/bin/pip install -r requirements-prep.txt
python3.12 -m venv .venv-score
.venv-score/bin/pip install -r requirements-score.txt
export PREP_PYTHON="$PWD/.venv-prep/bin/python"
export SCORE_PYTHON="$PWD/.venv-score/bin/python"
export GENERATION_PYTHON=/absolute/path/to/vllm-env/bin/python

"$PREP_PYTHON" configure.py --dataset-root /absolute/path/to/dataset-project --models-root /absolute/path/to/models --verification-python "$SCORE_PYTHON"
"$PREP_PYTHON" verify_models.py
"$PREP_PYTHON" prepare.py
bash align.sh
"$PREP_PYTHON" freeze.py
"$SCORE_PYTHON" validate.py
```

`configure.py` refuses an existing configured/frozen run. Model verification hashes every weight shard. Dataset preparation verifies Parquet checksums/counts and generates the fixed sample and all native prompts. Tokenizer alignment creates local metadata overlays for DeepSeek, AceReason and Phi-4, retaining native tokenizer files and templates, and verifies all 2,560 prepared prompts against the native JSON and vLLM tokenizer. `freeze.py` requires the model/tokenizer checks and hashes code, config and prepared inputs while excluding macOS metadata. `validate.py` checks grading, references, sample identity and expected response counts. Do not run Python with `-O`; assertions are intentional validation gates.

Do not edit a frozen run or regenerate its inputs. Use a fresh copy for a changed experiment. The generated manifest belongs to that specific configuration and cannot resume the historical cluster outputs after portability changes.

## Slurm generation

This launcher deliberately supports **one complete node with exactly eight A100 80GB GPUs**, eight independent replicas and batches of sixteen. It rejects partial allocations, larger nodes and 40GB devices. Select an eligible node from live scheduler state. Slurm GPU indices/ranges are mapped to NVIDIA UUIDs, not Linux device-minor numbers. Every allocated GPU must pass BF16, native-tokenizer, EOS-stopping and finite-logit checks before useful generation proceeds.

For the MadryLab high-priority queue, after preparation and validation pass:

```bash
export A100_80GB_NODE=your-confirmed-eight-gpu-node
sbatch --account=grad-students --partition=high-priority --qos=high-priority --nodes=1 --exclusive --gres=gpu:a100:8 --nodelist="$A100_80GB_NODE" --ntasks=1 --cpus-per-task=32 --mem=256G --array=0-4%1 --time=2-00:00:00 --requeue --job-name=reasoning-budget-eval --chdir="$PWD" --output="$PWD/slurm-%A_%a.log" --export=ALL --wrap="bash '$PWD/run.sh'"
```

Account/partition/QoS are site-specific. Array indices follow `experiment.json` model order. `%1` runs one model at a time. The exported `GENERATION_PYTHON` must be available on the compute node; `run.sh` does not activate an unrelated environment. No SSH credentials or job submission are performed by the Python code.

Each model has a process lock. Completed batch files are written atomically, checked against the immutable manifest and reused on restart. A worker failure stops sibling workers owned by this launcher. After generation, `run.py` invokes the configured verification Python for scoring and report generation. Intermediate reports can be partial until all models and reference reviews finish.

## Manual review and reports

Inspect flagged records in `outputs/MODEL/scored.jsonl`. Write `outputs/MODEL/manual-grades.json` as an object keyed by `QUESTION_KEY:BUDGET`. Each entry contains `correct` (boolean), `rationale` (nonempty string), and `completion_sha256` (SHA256 of the UTF-8 `completion` string). Review the final-answer section under the established metric; an intermediate formula inside unfinished reasoning is insufficient. Hash binding must match the exact reviewed response.

After adding reviews, rerun the original scoring pipeline under the scoring environment:

```bash
"$SCORE_PYTHON" score.py --model phi4
"$SCORE_PYTHON" summarize.py
```

Reports are `reports/RESULTS.md`, `results.csv`, `results.json` and `dashboard.html`. Per-model summaries bind their scored JSONL and manual-review file hashes. `complete=true` requires all five models and zero unresolved grades. Preserve old graded artifacts before replacing reports if you need an audit trail.

## Validation and source provenance

```bash
.venv-score/bin/python -m unittest discover -s . -p 'test_*.py' -v
bash -n run.sh align.sh
```

The CPU tests cover strict grading, incomplete reasoning, unsupported references, GPU-ID mapping, finished-only metrics, response-bound manual review, path configuration and manifest immutability. GPU startup checks remain required on a real allocation.

This is a reusable source export of the September 23 evaluation. Sampling, model parameters, native prompts and mathematical grading rules are preserved. Export changes make paths configurable, add an explicit freeze/weight-verification workflow, correct GPU-ID handling, bind manual reviews to completion hashes, and report finished-only length separately. The original completed experiment files are untouched. Generated questions, completions, weights, tokenizer copies, job records, local access configuration and credentials are not committed. The later distributed DeepSeek pass@8 continuation is a separate experiment.
