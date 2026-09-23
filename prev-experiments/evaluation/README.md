# Base versus GRPO math evaluation

This is a separate evaluation workflow. It compares each starting checkpoint with the final step-1000 checkpoint at staleness caps 2, 4, 6 and 8. It does not change training, export unfinished checkpoints, submit jobs or publish results.

## Model matrix

| Family | Starting model | Final staleness caps | Pinned final releases on September 19 |
| --- | --- | --- | --- |
| Qwen2.5-Math-1.5B | Pinned upstream revision | 2, 4, 6, 8 | 2, 4, 6, 8 |
| Qwen2.5-3B | Pinned upstream revision | 2, 4, 6, 8 | 2, 4, 8 |
| Qwen2.5-Math-7B | Pinned upstream revision | 2, 4, 6, 8 | 2 |
| Qwen3-14B | Pinned upstream revision | 2, 4, 6, 8 | Awaiting final exports |

There are 20 targets: four starting models and 16 trained models. `models.json` binds the four starting revisions, nine published final releases, tokenizer hashes and provenance references. Seven trained entries remain `awaiting_final`; preparation rejects them. This is an artifact inventory, not a live scheduler status report. Superseded staleness-zero snapshots are excluded.

“Base” means the model before our GRPO training. Qwen3-14B is already a post-trained hybrid reasoning model; Qwen2.5-3B is a general pretrained model. Report improvement within each family. Training GPU layouts changed across some staleness arms, so differences are not isolated causal effects of staleness.

## Frozen benchmark suite

| Benchmark | Problems | Role |
| --- | ---: | --- |
| MATH-500 | 500 | Continuity |
| AMC 2023 | 40 | Continuity |
| AIME 2024 / 2025 / 2026 | 30 each | Continuity and token-budget comparison |
| Minerva Math | 272 | Continuity |
| OlympiadBench | 675 | Text, final-answer subset |
| HMMT February 2025 | 30 | Added competition |
| HMMT November 2025 | 30 | Added competition |
| HMMT February 2026 | 33 | Added competition and token-budget comparison |
| BRUMO 2025 | 30 | Added competition |
| CMIMC 2025 | 40 | Added competition |

Total: **1,740 problems**. `benchmarks.json` pins every source revision and the five new parquet hashes. The retained 1,577 rows come from the exact frozen `eval.jsonl`, not a fresh reconstruction. Its bytes and the training data are checked against the recorded manifests. The seven continuity benchmarks appeared in training evaluations and are not previously unseen validation sets.

HMMT February 2026 really contains 33 rows; its last three prompts ask for proofs. All scores here measure final-answer correctness, not proof correctness. Complete comma-separated root lists remain one gold answer; they are not split into alternatives. Our frozen terminal-answer extraction is stricter than MathArena's publisher parser, so these are not claimed to reproduce leaderboard scores.

Before GPU evaluation, prepare an exact/normalized/near-duplicate audit against the actual 17,005 training rows. Confirmed matches are excluded from the filtered score; full-membership scores remain separately available. Near matches require documented keep/exclude decisions bound to the prompt hash. The audit also records duplicates across benchmarks. Do not average correlated duplicate questions into an overall score. This audit does not establish absence of pretraining overlap.

The added-only real audit on September 19 found **0 detected overlaps among 163 added questions** against the exact training file. All 163 also fit the Math models' 4,096-token context at the 3,072-output cap: the largest frozen Qwen2.5 prompt is 682 tokens, totaling 3,754. The full retained-file audit remains a cluster preparation step. Diagnostics are saved outside the repository under `tmp/math-sweep-20260919/added-overlap-audit.json` and `token-budget-check.json` in the workspace.

## Decoding configurations

| Profile | Benchmarks | Output-token limits | Samples/problem | Temperature / top-p / top-k | Default |
| --- | --- | --- | ---: | --- | --- |
| `greedy-native` | All 12 | 3,072; Minerva/Olympiad 2,048 | 1 | 0 / 1 / −1 | Yes |
| `sampled-native` | AIME and all five added sets | 3,072 | 16 | 0.6 / 1 / −1 | Yes |
| Same `sampled-native` profile | AIME 2024–26 and HMMT February 2026 | Additional 1,024 and 2,048 | 16 | 0.6 / 1 / −1 | Yes |
| `qwen3-long-nonthinking` | AIME 2024–26 and HMMT February 2026 | 8,192 / 16,384 | 16 | 0.6 / 1 / −1 | Opt in |
| `qwen3-long-thinking` | Same four sets | 8,192 / 16,384 | 16 | 0.6 / 0.95 / 20 | Opt in |
| `15b-extension-greedy` / `15b-extension-sampled` | AIME 2024–26 | 3,072 / 8,192 | 1 / 16 | 0 or 0.6 / 1 / −1 | Planning only |

The 3K sampled cells are shared by the main and budget comparisons; the planner does not duplicate them. The full default matrix has **560 cells and 194,480 responses**, with a maximum of 457,134,080 output tokens if every answer reaches its cap. These counts include pending models and precede overlap filtering. Start execution with data/model preparation and a GPU validation cell; measure throughput before assigning a sweep ETA.

Common settings: BF16, explicit stop IDs `[151643, 151645]`, min-p 0, repetition penalty 1, presence/frequency penalties 0, no text stop strings, `ignore_eos=False`, and `generation_config="vllm"`. Per-problem/sample seeds are stable across models and budgets. There are no tools or generated-code execution. The exact prompt is:

```text
Solve the following math problem. Explain your reasoning. End with either \boxed{...} or a final line `Final answer: ...`.

{problem}
```

All base/trained arms in a family share the frozen training tokenizer and chat template. Qwen3 continuity includes its explicit empty thinking block. The thinking diagnostic instead uses the pinned upstream Qwen3 template with thinking enabled. An unfinished thinking block cannot score as a final answer.

### Context and hardware

Native total context ceilings are 4,096 for the 1.5B/7B Math families and 32,768 for 3B/14B. The 14B ceiling is conservative relative to its config. The preparer counts rendered prompt tokens and rejects any prompt plus output cap that does not fit; it never truncates the prompt or silently reduces the budget. Dataset changes require an explicit new configuration, not per-model exclusions.

The initial native runner targets **one A100 80GB per cell**, tensor parallelism 1, 85% GPU memory reservation, 16 active sequences, 8,192 batched tokens, eager execution, chunked prefill and prefix caching disabled. Hardware and decoding settings are checked and recorded. Any change produces new prepared identities and must be matched across compared checkpoints. GPU memory and decoding gates still require actual allocated-GPU validation.

The 1.5B extension profiles are retained in the planner for the user-requested 8K experiment, but the new native runner intentionally rejects them. Both their 3K and 8K arms require the same 9,216-position experimental engine and the rotary-cache, long-prefill and boundary-generation checks established by `prime-workspace/experiments/aime-15b-staleness-length-v6` in the workspace. That historical run used eight samples; its results cannot fill these new 16-sample cells. Port and validate those gates before executing these profiles. No extension results are mixed into native comparisons.

## Commands

Run commands from the `staleness-analysis` root. Planning and report statistics use the Python standard library. Preparation and generation use the existing validated cluster runtime: Python 3.12, vLLM `0.26.0+cu129`, PyTorch `2.11.0+cu128`, Transformers `5.6.2`, verifiers `0.3.1`, math-verify `0.9.0`, latex2sympy2-extended `1.11.0`, SymPy `1.14.0`. The preparer verifies these versions and freezes additional tokenizer/data-library versions. Do not install a replacement runtime into a running training environment.

```bash
python evaluation/sweep.py plan --output outputs/math-sweep/plan.json
python evaluation/sweep.py prepare-data --output outputs/math-sweep/data.json
python evaluation/sweep.py validate-grader
python evaluation/sweep.py prepare --plan outputs/math-sweep/plan.json --cell CELL_ID_FROM_PLAN --data outputs/math-sweep/data.json --output-dir outputs/math-sweep/prepared
python evaluation/sweep.py run --prepared outputs/math-sweep/prepared/PREPARED_CELL_ID --output-root outputs/math-sweep/results
python evaluation/sweep.py report --prepared-root outputs/math-sweep/prepared --output-root outputs/math-sweep/results --report outputs/math-sweep/report.json
```

`prepare-data` supports `--retained-path`, `--training-path`, `--data-root` and `--decisions`; overrides still require exact file hashes. `prepare` supports `--tokenizer-root` for the cluster's experiment directory. Model snapshots are downloaded at immutable revisions into the configured Hugging Face cache. Use the user's persistent model/cache storage on the cluster.

To plan a subset, pass exact IDs with `--models` or profile IDs with `--profiles`. The default plan includes unavailable targets as pending. A prepared cell receives a new ID bound to actual inputs, runtime, implementation and model file hashes. No job is submitted by these commands. `run` requires an existing Slurm allocation and explicit GPU visibility. For eventual submission, follow the saved high-priority full-node policy unless the user overrides it; assign each allocated GPU to a cell worker, with disjoint visibility. The runner is one cell per process.

Overlap decisions use this schema; replace the identifiers and hash from the audit:

```json
{
  "benchmark:source_id": {
    "decision": "keep",
    "evidence": "Reviewed the candidate and documented why the mathematical tasks differ.",
    "prompt_sha256": "SHA256_FROM_AUDIT"
  }
}
```

## Outputs and checks

Preparation verifies published training cap, step 1,000, original base revision, training-data hash, export validation, all weight shards, tokenizer hashes, exact problem counts and grading fixtures. It saves token-ID prompts and refuses unresolved overlap candidates. Every GPU process first checks both EOS IDs and finite decoding.

Each cell stores `records.jsonl`, `probes.json`, probe-attempt history and `receipt.json`. Raw token IDs, raw text, decoded final-answer text, source/gold IDs, seeds, stop reasons, truncation, model identity and grader status are retained. A file lock prevents duplicate writers. Restarts validate saved identities and generate only missing samples. Torn JSON lines require explicit repair; they are never silently discarded. A completed cell must contain exactly the expected question/sample pairs.

Reports include greedy/sampled accuracy, pass@1/4/8/16 where supported, problem-level bootstrap 95% intervals, paired base deltas, lengths, truncation, terminal-syntax failures and measured GPU-hours. Pass@k means at least one correct response among k draws with an oracle selector; it is not single-answer accuracy. The frozen verifier converts some mathematical parsing failures/timeouts into an incorrect score, so the report does not claim to distinguish those causes. Fixed seeds do not guarantee identical token prefixes under different generation limits or batching; individual answer changes are not automatically evidence of more useful reasoning.

Reports require valid completion receipts. Missing or failed prepared cells require `--allow-partial`, and the output labels them explicitly. “Complete for prepared cells” does not mean all 20 targets are complete. Full benchmark membership and training-overlap-filtered scores are separate. Resume timing after a hard process interruption is a lower bound and is marked incomplete.

Run CPU verification from the repository root:

```bash
PYTHONPATH=evaluation python -m unittest discover -s evaluation/math_sweep -t evaluation -p 'test_*.py'
```

The scoring fixtures need the pinned verifier dependencies; a standard-library-only Python skips them. All 74 CPU tests passed in the local Python 3.12 verification environment, including the ten real mathematical-grader fixtures. The new multi-family runner has not yet been tested on an allocated GPU, and no benchmark results are claimed by this implementation.
