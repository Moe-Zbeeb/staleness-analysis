# DeepSeek 1.5B on-policy baseline

Job **2145261** was submitted on September 27, 2026 with account `grad-students`, partition `low-priority` and QoS `normal`. It requests one exclusive eight-A100 node, 128 CPU slots and a 45-day wall limit with automatic requeue disabled. It is a full 1,000-update run, independent of the queued k=256 job `2145258`. The user subsequently requested node 7. The pending job was held, moved to node 7 and released after updating and verifying the configuration manifest. All seven participating A100 80 GB GPUs passed the collective, BF16 backward, Flash Attention and vLLM checks; the seven-rank all-reduce produced 28 as required. The full-training supervisor has started. No completed optimizer update has been verified yet. See the [health receipt](../../diagnostics/onpolicy-node7-health-2145261.json).

The [study configuration](../../configs/deepseek15b-onpolicy-80gb-seed42-v1-opt.json) was copied from the optimized k=256 launch. The initial dictionary comparison confirmed only lag and output-path differences. The explicitly authorized node-7 override adds one hardware change: inference GPU count decreases from four to three. Lag changes from 256 to 0; all other study parameters remain identical. The original pending configuration is preserved in the cluster control directory under `before-node7/`. The [comparison receipt](../../diagnostics/deepseek15b-onpolicy-20260927.json) records the source config and frozen package hashes.

The pinned initial model is `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, with the same cleaned DeepScaleR dataset and grader. Each update uses 64 prompts × 8 responses, a 2,048-token prompt limit and 8,192-token response limit. GRPO, LR 1e-6, 30-step LR warmup, zero weight decay, seed 42, 1,000 updates and 100-update checkpoint interval are unchanged. Inference uses three GPUs, maximum 64 active sequences per GPU and dispatcher concurrency 256. Training uses four GPUs, activation checkpointing enabled, compilation disabled and forward resharding disabled.

With k=0, the controller generates one cohort with policy version t, trains once on that cohort, synchronizes policy t+1, then starts the next cohort. Every consumed rollout has age zero. There is no staleness bootstrap, deferred queue or extra deferred-generation pass. The optimizer's 30-step LR warmup remains enabled. This uses the same controller and imported PrimeRL library as the exact-staleness run; no training source change was needed.

## Launch and storage

The launcher first stages and verifies node-local source, runtime, model, tokenizer and dataset, then checks GPU health and seven-rank collectives before full training. It reserves the complete eight-GPU node and requires exactly seven working A100 80 GB GPUs, using the existing verified (8 allocated, 7 participating) exception. Four train and three generate; the unhealthy GPU is excluded by the probe. Previous thermal throttling on one working GPU remains a throughput limitation. The queued k=256 job still requires four training plus four inference GPUs on a healthy node. This topology difference must be recorded when comparing wall-clock performance.

- Launch control: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek15b-onpolicy-normal-20260927`
- Local workspace: `/tmp/staleness-storage-v2/mohamadzbib/deepseek15b-onpolicy-80gb-seed42-v1-opt`
- Local runtime: `/tmp/staleness-runtime/mohamadzbib/deepseek15b-onpolicy-v1`
- NFS backup: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/outputs/deepseek15b-onpolicy-80gb-seed42-v1-opt`
- XFS metrics: `/mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics/deepseek15b-onpolicy-80gb-seed42-v1-opt`

TensorBoard remains the only dashboard, with no intermediate evaluations. Background backup publication uses the existing checksum-verification process. The queued k=256 candidate's combined optimization still awaits full-pipeline validation; no production speedup is claimed for this baseline. Run times also include different overlap opportunities: k=0 generation must wait for each updated policy.

All 61 relevant queue, configuration and audit tests passed locally, including k=0 age, accounting and recovery cases. The cluster health checks subsequently passed; these checks do not establish completed training progress.

The [live startup review](../../diagnostics/onpolicy-live-review-2145261.json) subsequently confirmed inference readiness after 7m 16s of controller waiting, initial weight transfer and active response generation. At that snapshot there were still zero completed optimizer updates. Both shared destinations had matching sampled configuration/TensorBoard hashes. Two inference GPUs showed active thermal throttling, with one at 495 MHz, so current wall-clock performance should not be treated as a healthy-node optimization measurement.
