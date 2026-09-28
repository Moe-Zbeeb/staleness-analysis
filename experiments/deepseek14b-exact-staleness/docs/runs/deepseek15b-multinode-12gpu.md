# DeepSeek 1.5B exact k=256 on twelve GPUs

The user requested stopping the on-policy baseline and using the full twelve-A100 high-priority quota for one exact-k256 study. Replacement job **2145304** was admitted on `deep-chungus-3` and `deep-chungus-5`, with both partition and QoS `high-priority`, account `grad-students`, 96 CPUs, six GPUs and 200 GiB requested per node, a 45-day limit, and automatic requeue disabled. This allocation passed all twelve GPU checks, the cross-node collective, eight-engine routing and the first complete optimizer update. Update 1 and its metrics are present on both NFS and XFS.

| Node | Trainer GPUs | Inference GPUs |
| --- | ---: | ---: |
| deep-chungus-3 | 4 | 2 |
| deep-chungus-5 | 0 | 6 |
| Total | 4 | 8 |

The user explicitly allowed A100 40GB and 80GB GPUs. Both selected nodes expose A100 PCIe 40GB cards. Nodes 7 and 8 are excluded because of the known unhealthy device and unavailable node. No unrelated processes are killed by this launcher. Slurm applies its normal high-priority scheduling/preemption policy.

## Training contract

The model remains `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` on the pinned cleaned DeepScaleR dataset. The unchanged scientific settings include 64 prompts × 8 responses, 2,048 prompt tokens, 8,192 response tokens, 1,000 optimizer updates, GRPO clipping 0.2, centered advantages, global response-token normalization, LR 1e-6 with 30 warmup updates, zero weight decay and reference KL, gradient clipping 1, and no zero-advantage filtering. No intermediate evaluations or Runboard processes are enabled.

Updates 1–256 build history using on-policy updates and additional deferred cohorts. Updates 257–1,000 consume data exactly 256 optimizer updates old. The final queue-draining phase does not generate unused future cohorts. We did not reduce the lag, shorten responses, remove tokens, or skip optimizer updates.

The four trainer ranks stay together on one node. Inference uses eight independent TP=1 engines, 64 active sequences per engine and aggregate dispatcher concurrency 512. The `round_robin` router distributes requests across all eight engines; admin requests bypass the router so every engine joins each policy broadcast. Request retry count is one attempt, avoiding hidden replacement sampling. The run seed remains 42; independent engine seeds are 42–49 in endpoint order. Larger parallelism and routing change random-number consumption and floating-point execution; outputs are not promised to match the earlier topology bit for bit.

Training keeps activation checkpointing enabled, forward resharding disabled, FP32 optimizer/reductions, BF16 computation and learner compilation disabled. vLLM still performs its normal compilation and graph preparation. These choices use earlier component measurements; the combined twelve-GPU throughput has not yet been measured and is not claimed globally optimal.

## Implementation and validation

`multinode_run.py` stages a verified local runtime, model, tokenizer, dataset and frozen adapter on each node. It checks every allocated GPU, performs six-rank local BF16/FlashAttention/NCCL checks and a twelve-rank cross-node collective before starting inference. The collective must sum to 78. There is no silent reduction of GPU count.

The custom launcher accepts an explicit `RemoteInference` deployment manifest. It starts only four local trainer ranks, the existing exact-staleness controller and observers. Training, rollout transport, the deferred queue, broadcast handshakes, active metrics and full recovery checkpoints remain on the trainer node's local filesystem. Weights reach all eight engines using PrimeRL's NCCL transport, with the existing acknowledged policy-version barrier. Failure of a supervised inference process terminates the study instead of continuing with a smaller pool.

Official PrimeRL v0.9.0 at `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1` and vLLM source are unchanged. We added the official **vllm-router 0.2.0** wheel already pinned in PrimeRL's lockfile to each isolated local runtime. Its SHA256 is `bac193bedf10f9a0265fe4fdaae0f0418574cd1f15c45f27da1b4a2bae8c10b8`. The adapter records the router version in run identity; no shared environment is upgraded. P2P/SHM are disabled for the initial mixed-node PCIe-compatible NCCL setup, matching the earlier component benchmark environment; cross-node communication is measured at startup.

91 local tests passed covering configuration wiring, distinct engine seeds, invalid endpoint/device rejection, exact-age queue behavior, storage and recovery. A cluster config dry-run also resolved. Twelve GPUs and cross-node NCCL passed the runtime diagnostic: sum 78 and five 32 MiB broadcasts in 0.193373 seconds. The replacement subsequently completed a full-size update and policy broadcast. No speedup multiplier or completion ETA is established from this single update.

## Storage and recovery

Control: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek15b-k256-12gpu-hp-20260928-v6`.

Local workspace on both nodes: `/tmp/staleness-storage-v2/mohamadzbib/deepseek15b-exact256-12gpu-seed42-v6`. Runtime: `/tmp/staleness-runtime/mohamadzbib/deepseek15b-12gpu-v2`.

Full checkpoints remain every 100 updates. Verified background backups go to `outputs/deepseek15b-exact256-12gpu-seed42-v6` under the NFS project root; metrics and TensorBoard events also go to the matching XFS `metrics/` directory. Per-node health receipts, inference configs and termination logs are retained in the control directory. The deployment manifest is included in both shared metadata copies.

This initial multi-node launcher deliberately rejects `--resume`: a recovery launch must first validate deployment and checkpoint compatibility and stage the complete checkpoint on the trainer node. Checkpoint production is retained; unattended recovery has not been validated. Failed jobs are not automatically restarted from initial weights.

## Replaced jobs

On-policy job **2145261** was intentionally stopped through its supervisor; its saved run status records SIGTERM and no cleanup errors. Slurm labels this nonzero termination `FAILED`. Old queued eight-GPU k256 job **2145258** was cancelled. Twelve-GPU attempt **2145289** was cancelled during staging, before training, to give independent inference engines distinct RNG streams. All old outputs are retained.

See [submission evidence](../../diagnostics/multinode-12gpu-2145304.json) for the resolved study, storage specification, launch hashes and verified Slurm fields.

Network preflight in job **2145292** failed before inference/training because NCCL selected an unreachable IPv6 link-local address on `eth3`. The replacement derives the outbound IPv4 interface with `ip -j route get` for the peer node, sets `NCCL_SOCKET_FAMILY=AF_INET`, selects that exact interface, and records it per node. These are [documented NCCL settings](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html). The prior runtime is reused after its source checks; new source, assets, outputs and launch receipts have separate paths. No training update was lost in that failed preflight.

A second preflight, **2145293**, identified a Linux address-label mismatch: route devices are `eth0`/`eth3`, but their IPv4 labels exposed to `getifaddrs` are `enx1070fde5feb8`/`enx1070fde59b60`. The launcher now resolves the address label matching the outbound IPv4 source. A subsequent bounded diagnostic showed node3 selecting IB/RoCE and node5 selecting Socket, producing an incompatible NCCL handshake. Both sides now explicitly use `NCCL_NET=Socket` and `NCCL_IB_DISABLE=1`. The corrected twelve-rank diagnostic passed in **2145296**, which then released its gate into the same full-run allocation. There is no checkpoint or optimizer state to recover from the preceding attempts: they never entered training.

Job **2145296** passed the production network preflight and started the router, then stopped before training when the new backup command omitted `--destination`. The corrected command is exercised against the real backup CLI in a regression test. Job **2145301** uses a fresh output directory and the corrected immutable release. No committed optimizer update from the earlier attempts is discarded.

Job **2145301** passed GPU/network checks and synchronized the startup model to all eight engines, but the router initially registered six still-starting engines under `unknown` and sent generation only to the two ready engines. It was intentionally stopped with zero committed updates. The adapter now waits for every engine to serve the expected model, then requires all eight router entries to be healthy and correctly identified before starting training. The router uses `round_robin`: the previous load-based policy polled `/get_load`, which returns 404 in this vLLM version. This change distributes a 512-request cohort evenly without changing its prompts, sample count, or loss. See the [upstream readiness behavior](https://github.com/PrimeIntellect-ai/router/blob/v0.2.0/src/routers/http/router.rs) and [round-robin implementation](https://github.com/PrimeIntellect-ai/router/blob/v0.2.0/src/policies/round_robin.rs).

The corrected replacement is **2145304**, on the same twelve-GPU high-priority layout, with fresh outputs ending in `v6`.

## First completed update

Job **2145304** committed update 1 with 512 responses, bootstrap age 0, mean reward 0.4160, finite learner loss 0.00007294, entropy 0.8144, and mismatch KL 0.0004234. The deferred cohort generated at policy version 0 is scheduled for update 257. All eight inference engines served generation requests, and the updated policy was acknowledged before commit. NFS and XFS both contain the committed update and TensorBoard files; the backup process reports verified status.

The controller measured 883.02 seconds for the first bootstrap update: generation cohorts 134.12 and 133.61 seconds, learner wait 604.47 seconds, and weight transfer 9.78 seconds. These are first-update measurements, not steady-state estimates or a controlled comparison. Training is the dominant cost, so twelve allocated GPUs alone do not establish a speedup. The four trainer GPUs peaked around 11 GiB.

Two transient cross-node HTTP model-call errors were logged during the first cohort. The complete cohort subsequently passed the controller validation and update 1 completed, but this is not proof of an error-free transport or absence of internal SDK retries. Whole-episode and whole-agent retries remain disabled; the router allows one attempt. The attempt-level transport audit is a remaining limitation. No frozen library or scientific settings were changed to hide the errors.
