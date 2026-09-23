# Qwen2.5-3B staleness ≤6 Hugging Face release

Published publicly on 2026-09-19 at 22:41 UTC (2026-09-20 in Beirut).

| Field | Value |
| --- | --- |
| Model | [zbeeb/Qwen2.5-3B-GRPO-Staleness-6](https://huggingface.co/zbeeb/Qwen2.5-3B-GRPO-Staleness-6) |
| Immutable revision | [cfc1ce2e2924f86c3a62a3320c928e4f44f88759](https://huggingface.co/zbeeb/Qwen2.5-3B-GRPO-Staleness-6/tree/cfc1ce2e2924f86c3a62a3320c928e4f44f88759) |
| Collection | [Staleness GRPO models and data](https://huggingface.co/collections/zbeeb/staleness-grpo-models-and-data-6aaa9d1ea8997854e100da62) |
| Dataset | [zbeeb/Staleness-GRPO-DAPO-Math-17k](https://huggingface.co/datasets/zbeeb/Staleness-GRPO-DAPO-Math-17k) |
| Checkpoint | Final step 1000, cap 6, seed 42 |
| Training job | 2142032, deep-chungus-10, COMPLETED 0:0 |
| Release job | 2142063, deep-chungus-6, CPU only, COMPLETED 0:0 |
| Export | 3,085,938,688 parameters, original FP32, seven Safetensors shards |
| Public verification | 20 files, 12,355,310,150 bytes |

The original completion audit passed, all three completion markers matched the training manifest, and the release audit repeated the full training checks. The final recovery resumed full model, optimizer, scheduler and progress state at step 775 with two trainer GPUs and one inference GPU (TP1), following an earlier two-inference-GPU TP2 segment. This execution history is recorded in the model card and training configuration.

Export validation passed finite-weight checks, strict Transformers reload, tied-embedding checks, tokenizer round-trip and exact CPU probe-logit equality. Source checkpoint file sizes and modification times were unchanged. Anonymous verification checked all uploaded file sizes, weight SHA256 values and downloaded metadata hashes; an independent local download checked the metadata hashes again. The temporary release credential was removed.

The model includes the training tokenizer, generation settings, upstream license and modification notice, nine saved final training evaluation results, completion evidence and export checksums. Generation stops on token IDs 151645 and 151643; native positional settings are preserved. The reported scores are saved training-run results, with no new benchmark or 8K diagnostic in this release. Sampled AIME rows report mean accuracy across eight completions per problem.

Local scripts and receipts: `prime-workspace/experiments/staleness-grpo-release-3b6-20260920`.

Remote export: `/mnt/nfs/home/mohamadzbib/projects/models/staleness-grpo-release-3b6-20260920/Qwen2.5-3B-GRPO-Staleness-6`.
