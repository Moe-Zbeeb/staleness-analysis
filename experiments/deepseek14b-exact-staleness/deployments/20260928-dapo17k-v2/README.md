# DAPO 17k / DeepSeek 1.5B / exact k256: independent inference servers

The full study was submitted from source commit `cb27c5b`. Jobs 2145435 and 2145436 started on September 28, 2026. This is a fresh DAPO/6,144-token experiment; it does not resume the old DeepScaleR run or the zero-update v1 startup attempt.

| Role | Job | Node | High-priority allocation |
| --- | --- | --- | --- |
| Learner and current-policy generation | 2145435 | deep-chungus-5 | Full node, 9 A100 40GB, 96 CPUs; 4 trainers and 5 inference replicas |
| Historical bootstrap generation | 2145436 | deep-chungus-3 | 3 A100 40GB, 48 CPUs; inference only |

Both allocations explicitly use account `grad-students`, partition `high-priority`, QoS `high-priority`, and no automatic requeue. The exact effective commands and immediate scheduler verification are in [submission.json](submission.json). The 45-day learner and 14-day worker limits are allocation limits, not completion estimates. The worker exits when its 256 assigned historical cohorts are complete.

## Fixed experiment

The pinned model is `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`; the pinned dataset is `zbeeb/Staleness-GRPO-DAPO-Math-17k`, with all 17,005 questions retained. Each update consumes 64 prompts with eight responses each. Responses are capped at 6,144 tokens, with a 2,048-token prompt cap. The study performs 1,000 optimizer updates: 256 on-policy bootstrap updates followed by exact age-256 updates. The first stale update consumes responses from the initial policy while the learner is at policy version 256.

The GRPO objective, optimizer, sampling and remaining scientific settings are listed in [the protocol](../../docs/asynchronous-historical-rollouts.md). Zero-advantage tokens remain in the batch and normalization; their diagnostic fractions are logged. No intermediate evaluation or Runboard process runs. TensorBoard reads the recorded training journals.

## Communication and storage

All four trainer ranks stay on node 5. Each inference GPU runs an independent TP1/DP1 official PrimeRL server, behind its node's local router. There is no cross-node GPU collective. Weight updates target every local server explicitly. The pinned router permits one request attempt; it does not silently replace failed samples.

Node-local working directories are `/tmp/mohamadzbib-staleness-storage-v2/deepseek15b-dapo17k-6k-exact256-seed42-v2` and the same name with `-historical` for the worker. Runtime, assets, caches, active checkpoints and rollout journals are local. Immutable historical exports/results cross shared storage asynchronously; missing exact-policy data causes a deliberate learner wait.

Recovery checkpoints are saved at update 1, every 25 updates and the final update. The separate backup process verifies shared copies before publishing them. It retains the two latest verified recovery checkpoints and every 100-update milestone. NFS backup output is `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/outputs/deepseek15b-dapo17k-6k-exact256-seed42-v2`. Metric copies are under `/mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics/` with the same run name.

The frozen release and controls are under `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek15b-dapo17k-async-hp-20260928-v2`. Package and control hashes bind deployed files. Its model constants and active manifest select the pinned 1.5B model; the repository default remains 14B. Imported PrimeRL and vLLM source are unchanged.

## Validation

The source passed 629 CPU tests with one skipped, Ruff and `git diff --check`. This covers queue provenance, checkpoint integrity, verifier cases, independent GPU configuration, single-attempt routing and owned-process cleanup. Live GPU startup, the first optimizer update, a historical result and the first verified recovery checkpoint are separate deployment checks; scheduler `RUNNING` alone does not establish them.

September 28 11:14 UTC check: both jobs were running. All nine main-node GPUs passed individual probes, all-reduce sum 45, BF16 backward, Flash Attention backward and vLLM RMS normalization. The learner independently checked all 17,005 dataset rows, the manifest and the clean official checkout. Its five inference replicas completed startup, synchronized the initial weights and began generating and grading the first batch, including positive rewards. No complete batch, optimizer update or recovery checkpoint was committed yet. The remote worker was still staging its runtime.

The historical queue contained the verified publication metadata and five weight files for policy zero, designated for update 257; no result or failure marker existed yet. NFS and XFS metric copies matched byte-for-byte in the sampled snapshot, TensorBoard copies had equal sizes, and the separate backup process reported verified status. That startup backup status does not establish checkpoint completeness. Main-node local storage had about 1,320 GiB free; NFS about 17 TiB and XFS about 2.4 TiB free.
