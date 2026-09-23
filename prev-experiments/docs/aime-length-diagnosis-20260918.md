# AIME length diagnostic: 1.5B staleness ≤2

Checked: 2026-09-17T23:47:11.693219+00:00

## Finding

The reported trained-model 8K score of zero is invalid as a measure of model quality. Job 2141853 completed, but its evaluation configuration had two defects. No fresh inference was run during this diagnosis.

1. **Missing stop token.** The trained model emitted token 151643 (`<|endoftext|>`) in 756/810 answers, but generation continued. The explicit stop list only included 151645. Base model metadata declares EOS 151643; the exported trained artifact declares EOS 151645. The base run stopped at 151643; the trained run did not. The exact internal source of this effective arm difference is not fully traced, so a corrected harness must explicitly handle both IDs and record the effective configuration.
2. **Invalid context extension.** The harness allowed 9,216 total tokens using `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, while both model configurations retained `max_position_embeddings=4096`. Installed vLLM 0.26.0 Qwen2 code sizes the rotary-position cache from that configuration. All 952 outputs that developed a repeated tail—810 trained and 142 base—began repeating token 1023 (`lock`) at prompt length plus output offset 4097. Raising the allowed length did not correctly extend the positional cache. Logits/NaNs were not inspected.

The strict final-answer grader then scored the repeated tails as incorrect, even where a correct answer had appeared before EOS. Checkpoint files passed their recorded hash checks; this diagnosis found no evidence of weight corruption.

## Results: sampled mean accuracy

Each cell uses 30 questions × 8 samples = 240 answers, temperature 0.6. These percentages are mean sampled accuracy (pass@1), not pass@8. Historical 3K cells and the new diagnostic differ in engine/batching/RNG execution; this is not a controlled token-budget comparison.

| Set | Historical base, 3,072 | Historical trained, 3,072 | Trained saved answers rescored at first EOS |
|---|---:|---:|---:|
| AIME24 | 2.92% | 10.00% | 12.08% (29/240) |
| AIME25 | 3.75% | 6.67% | 6.25% (15/240) |
| AIME26 | 2.92% | 7.08% | 5.42% (13/240) |
| Aggregate | 3.19% (23/720) | 7.92% (57/720) | 7.92% (57/720) |

## Results: greedy

| Set | Historical base, 3,072 | Historical trained, 3,072 | Trained saved answers rescored at first EOS |
|---|---:|---:|---:|
| AIME24 | 3.33% | 16.67% | 20.00% (6/30) |
| AIME25 | 3.33% | 6.67% | 10.00% (3/30) |
| AIME26 | Not recorded | Not recorded | 16.67% (5/30) |

Recovered trained greedy total: 14/90 = 15.56%. Recovered trained sampled pass@8: AIME24 10/30 = 33.33%; AIME25 3/30 = 10%; AIME26 4/30 = 13.33%.

**All 71 recovered correct answers ended before 3,072 output tokens** (maximum 1,971 output tokens to first EOS). This run supplies no evidence of a correct answer obtained from extra reasoning beyond the 3K cap. It also cannot establish that correctly implemented 8K would not help.

## Recovery method and limits

Read the original JSONL with standard-library Python, split saved `raw_completion` at the first `<|endoftext|>` and checked its presence against token ID 151643. Applied the original `exact_math.py` terminal-answer extraction functions and the repository boxed-answer extractor, with exact integer comparison (leading zeros ignored; the single degree-marked gold value was normalized). Fifteen noninteger candidates were separately reviewed and all were unequal to their gold answer; symbolic expressions with unresolved variables are not numeric answers. Full tokenizer/symbolic-grader runs timed out during shared-filesystem access, so these results are an explicitly documented offline recovery, not a completed rerun of the original full grader.

The 54 trained answers without EOS were retained as failures in the denominator; their long-context generation was invalid. The recovered scores are therefore not valid 8K benchmark cells. Original files and original Comet scores were preserved; the zero-valued Comet cells remain uncorrected and must be treated as invalid.

## Required corrected experiment

Use explicit stopping for 151643 and 151645. Validate a separately versioned context-extension configuration by inspecting the actual positional cache and exercising generation past the native boundary before a full sweep. Record the extension method; this is still experimental context extension. Run fresh matched 3,072 and 8,192 arms for base and trained models with the same prompts, grading, sampling seeds and settings, and report truncation and stopping behavior. Do not modify the frozen executed experiment or overwrite its raw results.

## Provenance

- Model: `zbeeb/Qwen2.5-Math-1.5B-GRPO-Staleness-2`, final step 1000.
- Evaluation job: `2141853`, completed exit `0:0`, deep-chungus-11, two normal-priority A100 GPUs, elapsed 1:03:00.
- Remote raw directory: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/outputs/aime-length-qwen15b-stale2/20260918`.
- Frozen evaluation: `prime-workspace/experiments/aime-length-qwen15b-stale2/evaluate.py`.
- Local evidence directory: `tmp/aime-diagnosis-20260918/`.
- `raw-analysis.json`: token-level audit of all 1,620 responses.
- `runtime_source.json` and `stop_source.json`: installed source excerpts.
- `prefixes.json`: all 810 trained response prefixes and gold answers.
- `rescore.json`: per-response recovered scores, aggregate cells, review notes and examples.
- `rescore_local.py`: reproducible extraction and integer-comparison script; its 15 noninteger candidates are adjudicated in `rescore.json`.
- `evidence-sha256.json`: hashes of local evidence files.
- Historical results: `tmp/aime-length-15b-20260918/summary.json`.
