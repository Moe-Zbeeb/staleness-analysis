# Qwen2.5-3B: node-local exact-256 study

**Historical attempt: job 2145176 failed before producing rollouts or completing updates.** Its long temporary path exceeded the Unix-domain socket limit during vLLM startup. All outputs and verified shared copies are preserved. See the [replacement launch](qwen25-3b-local-v5.md).

Job **2145176** replaced the shared-storage startup **2145148**, which was cancelled without a committed optimizer update. The original output and controls are preserved. The replacement passed input/runtime verification, seven-GPU collective and backward checks, and launched the model workers. TensorBoard and the separate backup process are running. Initial NFS and XFS copies were independently checksum-checked; no completed update or full-size checkpoint is claimed yet.

The model remains `Qwen/Qwen2.5-3B` at `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`. The configuration remains 1,000 total updates, lag 256 after 256 bootstrap updates, 64 prompts × 8 responses, 2,048 prompt tokens, 8,192 response tokens, seed 42 and zero weight decay. The same 37,696 questions, 17 exclusions, grader, tokenizer and delayed-cohort algorithm are retained. There are no intermediate evaluations. Checkpoints remain every **100** updates, as requested; the teammate's 25-update checkpoint schedule and historical-policy-generation algorithm were not adopted.

The job requests the complete eight-A100-80GB node `deep-chungus-7`, 128 CPUs, all node memory, high-priority partition/QoS, the `grad-students` account, a 45-day limit and no automatic requeue. Four trainer GPUs plus three inference GPUs must pass health checks; the previously authorized eighth-card hardware exception remains explicit. A pending attempt, **2145168**, was cancelled before execution because the submission guard initially treated Slurm's `NumNodes=1-1` representation as a mismatch. The corrected guard verifies that both ends of the range are one.

| Component | Location |
| --- | --- |
| Active workspace | `/tmp/staleness-storage-v2/mohamadzbib/qwen25-3b-exact256-80gb-seed42-v4-local` |
| Runtime | `/tmp/staleness-runtime/mohamadzbib/qwen25-3b-local-v1` |
| Full shared backup | `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/outputs/qwen25-3b-exact256-80gb-seed42-v4-local` |
| Metrics and TensorBoard mirror | `/mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics/qwen25-3b-exact256-80gb-seed42-v4-local` |
| Launch controls | `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/qwen25-3b-full-hp-local-20260927` |
| Scheduler logs | The launch-control path with `-logs-r3` appended |

The first local attempt, **2145169**, passed all seven GPU health checks but failed before output creation: a broad rsync exclusion omitted three tracked monitor files, and missing Git LFS filters made hydrated fixtures appear dirty. The exact official files were restored, exclusions anchored to the checkout root, and checksum-pinned Git LFS configured in local repository metadata. The original integrity guard now passes before GPU startup. Failed attempt files are preserved, and retry logs/stop markers are isolated by job ID.

Source **fd6486c** adds storage orchestration around the frozen scientific release. Official PrimeRL and imported library source remain unchanged. The copied virtual environment's interpreter links, editable-import paths and entry-point paths are relocated; unused package tests and bytecode caches are omitted. CPython, PrimeRL (including namespace package locations), Torch, vLLM and Verifiers were verified to resolve locally. No `LD_LIBRARY_PATH`, CUDA-home or Conda override was present. The namespace-package audit was corrected during preparation before launching training.

All model files match the original pinned manifest. The dataset hash is `ebaf012ab811d4569e1409438daedac04bb671569d52a47bfb2d512ea784b0ff`; the prepared tokenizer hash is unchanged and all five native-template/token-ID probes pass. The data manifest retains content identity `abbeceb7c9b1904f739ea6876db28d23ba40fc97316f154f643d062fc3e13340`.

The local runtime copy required substantial one-time NFS reads. After switching to eight bounded package transfers, that phase completed in 1,124 seconds, reusing earlier copied files. This is preparation timing, not a training-speed measurement or a controlled storage benchmark.

The [storage guide](../node-local-storage.md) explains background copying, checksum verification, atomic checkpoint publication, failure behavior and recovery limits. Training uses TensorBoard only. Shared destinations are written by a separate backup process; a checkpoint is a durable recovery point only after its published `backup-verified.json` exists.

Validation: **34 focused tests passed**, Ruff and diff checks passed, all staged input checks passed, and local runtime imports passed. The tests include restoring exact queued cohorts from a verified shared checkpoint and ensuring interrupted transfers never publish a complete checkpoint. The live seven-GPU test returned all-reduce 28 on every rank; BF16 backward, FlashAttention backward and vLLM RMSNorm passed on all seven cards. Initial backup inventories contain 81 NFS files and 71 XFS files, including matching run identity and TensorBoard event checksums. Available storage was approximately 1,461 GiB local, 13,400 GiB NFS and 2,617 GiB XFS. Actual first-update throughput and the first 100-update recovery checkpoint remain unverified.

Machine-readable submission, configuration hashes and validation evidence are in [the archived migration record](../../diagnostics/node-local-v4-20260927.json).
