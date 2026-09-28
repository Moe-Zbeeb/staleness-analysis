# Qwen2.5 base 3B, DAPO17k, exact k256, batch 32

This additional full study runs alongside the unchanged DeepSeek 1.5B jobs 2145464 and 2145465. It starts from pinned Qwen/Qwen2.5-3B base revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`, with a fresh optimizer and rollout queue.

| Role | Job | Node | GPU layout | Priority |
| --- | --- | --- | --- | --- |
| Learner and current inference | 2145486 | deep-chungus-7 | 4 trainer + 3 inference, A100 80GB | Normal |
| Bootstrap historical inference | 2145487 | deep-chungus-1 | 3 inference, A100 40GB | Normal |

Both allocations started September 28 at approximately 13:13 UTC. Node 7 is exclusive. Its preparation job 2145485 verified seven healthy GPUs; device 7 timed out during CUDA initialization. The user previously authorized seven healthy GPUs on this node. The full job requests seven, and repeats allocation-bound device and collective validation before training. Slurm RUNNING alone does not establish successful training startup. The learner has a 45-day limit and the historical worker 14 days; automatic requeue is disabled. The existing 1.5B allocation occupies the 12-GPU high-priority quota, so these jobs use the authorized normal-priority fallback.

## Study

The global batch is 32 responses: four questions with eight responses each. The run targets 1,000 optimizer updates, exact lag 256, a response cap of 6,144 tokens and prompt cap of 2,048 tokens. DAPO17k source bytes and included question order match the 1.5B run: all 17,005 questions passed preparation, none were excluded. The tokenizer-specific manifest is regenerated, not copied from DeepSeek.

GRPO settings remain LR 1e-6, 30-step warmup, clip 0.2, centered group advantages, global token-mean loss, zero weight decay and zero reference KL. Seed 42, microbatch one packed sequence, BF16 compute, FP32 optimizer/reduction, activation checkpointing, disabled compilation and disabled forward resharding remain unchanged. See [all configuration differences](configuration-diff.json), including GPU count, ports and isolated paths.

Updates 1–256 train on-policy while the remote worker produces separate future cohorts from immutable policy versions 0–255. Update 257 consumes policy-0 rollouts at learner version 256. Subsequent consumed cohorts must have exact age 256; unavailable cohorts cause a wait. After bootstrap, local inference produces future cohorts while training consumes older ones. The historical worker exits after its bootstrap workload. The final 256 updates drain the queue.

## Model-specific changes and limitations

Qwen2.5-3B is a base model, not DeepSeek-R1 and not Qwen2.5-3B-Instruct. It does not promise native `<think>` output. Accordingly `reasoning_required=false`; the prompt still requests step-by-step reasoning and a boxed answer. Earlier runs of this base model had zero-reward cohorts. This is a scientifically material model/format difference, so equal optimizer settings do not guarantee comparable learning signal or speed.

The frozen application release derives from commit `cb27c5b`. Only `src/deepseek_study/__init__.py` (model ID/revision), `dataset/assets.py` (validate against Qwen's original tokenizer and native BOS/EOS instead of DeepSeek-specific think/BOS/EOS assumptions), and `manifests/model.json` differ from the active 1.5B release. Their deployed copies and package hashes are recorded here. Imported official PrimeRL and vLLM source are unchanged. No reward parser or GRPO loss changes were made for this deployment.

## Persistence

Active runtime, assets, caches, policies and recovery checkpoints use node-local storage. Checkpoints are saved at update 1, every 25 updates and completion, then checksum-verified by a separate NFS backup process. Metrics are mirrored to XFS and logged to TensorBoard. Runboard and intermediate evaluation remain disabled. This run has unique output, history, runtime and port paths; it does not reuse another run's checkpoints or rollouts.

Cluster controls: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/qwen25-3b-dapo17k-async-b32-normal-20260928-v1`.

NFS output: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/outputs/qwen25-3b-dapo17k-6k-exact256-b32-seed42-v1`.

The [submission receipt](submission.json) records exact commands and initial scheduler fields. Initial PENDING entries are submission-time snapshots, not current status. Preparation evidence is in [GPU probes](gpu-probes.json) and [preflight](preflight.json). No full-run duration estimate is established for this model/configuration yet.
