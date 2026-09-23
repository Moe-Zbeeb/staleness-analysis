# AIME comparison: all Qwen2.5-Math-1.5B versions

Requested September 18, 2026. Compare the pinned base model and final step-1000 GRPO models with staleness caps 2, 4, 6 and 8. Run on two GPUs of deep-chungus-11 at normal priority, continuing the user's explicit scheduling instruction for this AIME evaluation.

## Controlled comparison

| Model | Checkpoint | Output budgets | Benchmarks |
|---|---|---|---|
| Qwen2.5-Math-1.5B base | Pinned original revision | 3,072 and 8,192 | AIME 2024, 2025, 2026 |
| GRPO staleness ≤2 | Step 1000 | 3,072 and 8,192 | AIME 2024, 2025, 2026 |
| GRPO staleness ≤4 | Step 1000 | 3,072 and 8,192 | AIME 2024, 2025, 2026 |
| GRPO staleness ≤6 | Step 1000 | 3,072 and 8,192 | AIME 2024, 2025, 2026 |
| GRPO staleness ≤8 | Step 1000 | 3,072 and 8,192 | AIME 2024, 2025, 2026 |

Prompts contain 88–450 tokens. Gold normalization changes seven zero-padded integer strings without changing their values and strips the degree annotation from AIME2025-II-4. Original labels remain in the saved inputs and responses.

Each model and budget produces 90 greedy answers and 720 sampled answers (eight samples per question, temperature 0.6). Total: 8,100 scored responses. The report separates greedy accuracy, mean sampled accuracy and pass@8.

Within a model, only the configured output-token limit differs. Prompts, tokenization, seeds, temperature, top-p/top-k, stopping, precision, batching limits, grading and data are fixed. Both arms use a 9,216-position unscaled context extension. This is experimental extrapolation beyond the native 4,096-token context. Historical 3K and invalid 8K outputs are excluded from these fresh matched comparisons. Dynamic batching can still change numerical execution; shared-prefix agreement is measured rather than assumed.

## Validation and provenance

The job validates all model export hashes and final step markers; checks all model tokenizers against the shared prompt IDs; stages and rehashes model files; tests terminal-answer grading; audits the actual rotary position caches through position 9,215; forces and verifies both EOS IDs (151643 and 151645); and checks finite generation beyond the native context boundary. Probe responses are excluded from scores.

Raw response records retain prompt hashes, seeds, gold answers, raw text, decoded text, generated token IDs, extracted terminal answers, reward, stopping reason and truncation. Completion receipts prevent reuse of incomplete cells. The final summary includes per-year 8K scores, matched 3K/8K overall results, differences, parsing and truncation, plus separate greedy results. Model staleness comparisons also reflect the original training configurations; this evaluation does not make them a controlled training ablation.

## Attempts

- Job 2141896 (v2) failed the GPU UUID allocation check before model preparation or evaluation. Slurm's device minor numbers did not select the same physical GPUs through CUDA ordinal interpretation. No benchmark responses were produced.
- Job 2141897 (v3) preserves the failed attempt and translates Slurm's allocated device minors through `/proc/driver/nvidia/gpus/*/information` into explicit GPU UUIDs before CUDA initialization. It reruns the same strict UUID, BF16 and NCCL checks before evaluation.

- Job 2141897 passed both GPU health checks but stopped at the gold-label regression test before model staging. The historical verifier rejects an integer response against a degree-annotated gold label.
- Job 2141898 (v4) adds explicit integer normalization for all AIME gold labels, preserving originals. Inspection found one noncanonical label: AIME2025-II-4, `336^\circ`. The same normalized golds are used across every new cell.

- Job 2141898 (v4) passed model/data preparation but both vLLM workers failed before generation: allocated GPUs had 28–29 GiB free, below the configured 55.48 GiB reservation. No scored responses were produced.
- Job 2141901 (v5) reserves 25% of each GPU (about 19.8 GiB), identically across every cell. All other decoding settings are preserved. Node-local model files are reused and rehashed.

- Job 2141901 (v5) loaded both models and allocated enough KV cache, but the cache-audit callback was blocked by vLLM function-serialization restrictions before scoring.
- The v6 harness uses a named worker extension and string RPC to inspect the actual cache, retaining the default serialization restriction. It also verifies each engine worker GPU UUID against the allocated device.

## Locations

- Local implementation: `prime-workspace/experiments/aime-15b-staleness-length-v6`
- Remote implementation: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/experiments/aime-15b-staleness-length-v6`
- Remote output: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/outputs/aime-15b-staleness-length-v6/20260918`
- Local submission receipt and inspections: `tmp/aime-all-15b-v6-20260918`
- Comet project after successful aggregation: `qwen25-math15b-aime-controlled-length-v6`

Scores remain pending until the validation gates and completed-cell checks pass.

## Verified startup at 13:30 UTC

Job 2141903 is generating scored responses. Base and staleness ≤2 have passed exact GPU UUID checks, the actual 9,216 × 128 rotary cache audit (including positions 4,096, 8,191 and 9,215), both EOS tests, 9,000-token prefill plus 128 generated tokens, and 4,300-token forced generation across the native context boundary with finite selected log probabilities. First saved batches contain 32 responses per model, with no post-EOS tokens. Other models repeat the same gates before evaluation. Final scores remain pending.

Initial raw answers sometimes contain model-generated Python blocks, fabricated-looking output blocks, and the text `Reach max function call limit.` These are model-generated text, not tools executed by this evaluator. They are preserved; only the terminal answer is graded. This can cause nonanswers independently of token truncation and should be distinguished in the final diagnosis.

Separately, 7B staleness ≤6 was submitted as job 2141902 at the user-authorized normal priority, requesting all nine GPUs on deep-chungus-5 (8 trainer + 1 inference). Its frozen and resolved configs passed validation. It is pending resources because jobs 2141899 and 2141900 occupied the node before submission.
