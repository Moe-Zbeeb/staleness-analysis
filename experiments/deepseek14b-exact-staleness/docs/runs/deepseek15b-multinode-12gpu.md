# DeepSeek 1.5B exact k=256 on twelve GPUs

The user requested stopping the on-policy baseline and using the full twelve-A100 high-priority quota for one exact-k256 study. Replacement job **2145292** was admitted on `deep-chungus-3` and `deep-chungus-5`, with both partition and QoS `high-priority`, account `grad-students`, 96 CPUs, six GPUs and 200 GiB requested per node, a 45-day limit, and automatic requeue disabled. Startup validation is pending; allocation is not evidence of completed training.

| Node | Trainer GPUs | Inference GPUs |
| --- | ---: | ---: |
| deep-chungus-3 | 4 | 2 |
| deep-chungus-5 | 0 | 6 |
| Total | 4 | 8 |

The user explicitly allowed A100 40GB and 80GB GPUs. Nodes 7 and 8 are excluded because of the known unhealthy device and unavailable node. No unrelated processes are killed by this launcher. Slurm applies its normal high-priority scheduling/preemption policy.

## Training contract

The model remains `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` on the pinned cleaned DeepScaleR dataset. The unchanged scientific settings include 64 prompts × 8 responses, 2,048 prompt tokens, 8,192 response tokens, 1,000 optimizer updates, GRPO clipping 0.2, centered advantages, global response-token normalization, LR 1e-6 with 30 warmup updates, zero weight decay and reference KL, gradient clipping 1, and no zero-advantage filtering. No intermediate evaluations or Runboard processes are enabled.

Updates 1–256 build history using on-policy updates and additional deferred cohorts. Updates 257–1,000 consume data exactly 256 optimizer updates old. The final queue-draining phase does not generate unused future cohorts. We did not reduce the lag, shorten responses, remove tokens, or skip optimizer updates.

The four trainer ranks stay together on one node. Inference uses eight independent TP=1 engines, 64 active sequences per engine and aggregate dispatcher concurrency 512. The load-aware `power_of_two` router fronts all engines; admin requests bypass the router so every engine joins each policy broadcast. Request retry count is one attempt, avoiding hidden replacement sampling. The run seed remains 42; independent engine seeds are 42–49 in endpoint order. Larger parallelism and routing change random-number consumption and floating-point execution; outputs are not promised to match the earlier topology bit for bit.

Training keeps activation checkpointing enabled, forward resharding disabled, FP32 optimizer/reductions, BF16 computation and learner compilation disabled. vLLM still performs its normal compilation and graph preparation. These choices use earlier component measurements; the combined twelve-GPU throughput has not yet been measured and is not claimed globally optimal.

## Implementation and validation

`multinode_run.py` stages a verified local runtime, model, tokenizer, dataset and frozen adapter on each node. It checks every allocated GPU, performs six-rank local BF16/FlashAttention/NCCL checks and a twelve-rank cross-node collective before starting inference. The collective must sum to 78. There is no silent reduction of GPU count.

The custom launcher accepts an explicit `RemoteInference` deployment manifest. It starts only four local trainer ranks, the existing exact-staleness controller and observers. Training, rollout transport, the deferred queue, broadcast handshakes, active metrics and full recovery checkpoints remain on the trainer node's local filesystem. Weights reach all eight engines using PrimeRL's NCCL transport, with the existing acknowledged policy-version barrier. Failure of a supervised inference process terminates the study instead of continuing with a smaller pool.

Official PrimeRL v0.9.0 at `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1` and vLLM source are unchanged. We added the official **vllm-router 0.2.0** wheel already pinned in PrimeRL's lockfile to each isolated local runtime. Its SHA256 is `bac193bedf10f9a0265fe4fdaae0f0418574cd1f15c45f27da1b4a2bae8c10b8`. The adapter records the router version in run identity; no shared environment is upgraded. P2P/SHM are disabled for the initial mixed-node PCIe-compatible NCCL setup, matching the earlier component benchmark environment; cross-node communication is measured at startup.

86 local tests passed covering configuration wiring, distinct engine seeds, invalid endpoint/device rejection, exact-age queue behavior, storage and recovery. A cluster config dry-run also resolved. Actual GPU startup, cross-node weight updates and full-size training remain separate runtime gates. No speedup multiplier or completion ETA is established yet.

## Storage and recovery

Control: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek15b-k256-12gpu-hp-20260928-v2`.

Local workspace on both nodes: `/tmp/staleness-storage-v2/mohamadzbib/deepseek15b-exact256-12gpu-seed42-v2`. Runtime: `/tmp/staleness-runtime/mohamadzbib/deepseek15b-12gpu-v2`.

Full checkpoints remain every 100 updates. Verified background backups go to `outputs/deepseek15b-exact256-12gpu-seed42-v2` under the NFS project root; metrics and TensorBoard events also go to the matching XFS `metrics/` directory. Per-node health receipts, inference configs and termination logs are retained in the control directory. The deployment manifest is included in both shared metadata copies.

This initial multi-node launcher deliberately rejects `--resume`: a recovery launch must first validate deployment and checkpoint compatibility and stage the complete checkpoint on the trainer node. Checkpoint production is retained; unattended recovery has not been validated. Failed jobs are not automatically restarted from initial weights.

## Replaced jobs

On-policy job **2145261** was intentionally stopped through its supervisor; its saved run status records SIGTERM and no cleanup errors. Slurm labels this nonzero termination `FAILED`. Old queued eight-GPU k256 job **2145258** was cancelled. Twelve-GPU attempt **2145289** was cancelled during staging, before training, to give independent inference engines distinct RNG streams. All old outputs are retained.

See [submission evidence](../../diagnostics/multinode-12gpu-2145292.json) for the resolved study, storage specification, launch hashes and verified Slurm fields.
