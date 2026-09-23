# Qwen2.5-3B staleness ≤8 Hugging Face release

Published publicly and verified on 2026-09-19 at 02:11 UTC.

| Field | Value |
| --- | --- |
| Model | [zbeeb/Qwen2.5-3B-GRPO-Staleness-8](https://huggingface.co/zbeeb/Qwen2.5-3B-GRPO-Staleness-8) |
| Immutable revision | [42366582c633b1ab7e5e8df78d7fc18ab2c47870](https://huggingface.co/zbeeb/Qwen2.5-3B-GRPO-Staleness-8/tree/42366582c633b1ab7e5e8df78d7fc18ab2c47870) |
| Collection | [Staleness GRPO models and data](https://huggingface.co/collections/zbeeb/staleness-grpo-models-and-data-6aaa9d1ea8997854e100da62) |
| Training dataset | [zbeeb/Staleness-GRPO-DAPO-Math-17k](https://huggingface.co/datasets/zbeeb/Staleness-GRPO-DAPO-Math-17k) |
| Training checkpoint | Step 1000, staleness cap 8, seed 42 |
| Training job | 2141858, deep-chungus-7 |
| Release job | 2141985, deep-chungus-6, CPU only, COMPLETED 0:0 |
| Export | 3,085,938,688 parameters, original FP32 precision, seven Safetensors shards |
| Public verification | 20 files, 12,355,305,383 bytes |

The release contains model weights, training tokenizer and chat template, generation settings, upstream license and modification notice, training configuration, nine final training evaluation results, completion verification, and an export manifest with SHA256 hashes. Both the model and dataset were verified in the public Collection.

## Completion audit

Training reached 1,000 updates and completed all nine final evaluations with zero recorded evaluation errors. The original job subsequently failed its post-training audit because the old step-725 resume checkpoint had been removed by the configured retention policy.

A separate release audit verified the successful resume using pinned historical evidence and retained newer checkpoints, then checked all 1,000 steps, policy ages, finite metrics, final evaluations and the paired step-1000 checkpoint. Original training manifests, checkpoint files and Slurm failure history were preserved. The full evidence is in [completion-verification.json](https://huggingface.co/zbeeb/Qwen2.5-3B-GRPO-Staleness-8/blob/42366582c633b1ab7e5e8df78d7fc18ab2c47870/completion-verification.json).

## Export verification

The export passed finite-weight checks, strict Transformers reload, tied-embedding checks, tokenizer round-trip, and exact equality of CPU probe logits before and after serialization. Source checkpoint file sizes and modification times were unchanged. Anonymous public verification checked file sizes, Safetensors SHA256 values and downloaded metadata hashes. Local metadata hashes independently matched the export manifest after downloading the immutable revision.

Generation stops on token IDs 151645 and 151643. Native positional settings are preserved. Published scores are the saved training-run evaluations; this release did not run a fresh benchmark or an 8K diagnostic. Sampled AIME scores are mean accuracy over eight completions per problem, not pass@8.

Local release scripts, specification, assets and publication receipts are saved in `prime-workspace/experiments/staleness-grpo-release-3b8-20260919`. The remote export is `/mnt/nfs/home/mohamadzbib/projects/models/staleness-grpo-release-3b8-20260919/Qwen2.5-3B-GRPO-Staleness-8`.
