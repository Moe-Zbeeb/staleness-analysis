# DeepSeek-R1-Distill-Qwen-7B high-priority studies

The user requested full k256 and k0 studies with the same DAPO17k dataset, 32 responses per update (four questions times eight responses), 1,000 updates, 6,144 response tokens, 2,048 prompt tokens, seed 42 and unchanged GRPO/optimizer settings. Both training jobs must use high-priority partition and QoS. The k256 learner must run on A100 80GB GPUs with a separate bootstrap historical-inference pool.

The user subsequently requested both studies queued in high priority, on A100 80GB GPUs, expecting an 18-GPU allowance. Slurm still reports a per-user high-priority limit of 12 A100 GPUs and a QoS-wide limit of eight running jobs. The submissions request 18 GPUs in total; they cannot all run concurrently under the observed 12-GPU limit. Existing 1.5B runs were preserved. No fallback to another QoS is configured.

| Role | Job | GPUs | Placement |
| --- | --- | --- | --- |
| k256 learner and current-policy inference | 2145688 | 4 trainer + 4 inference | One exclusive eight-GPU node |
| k256 bootstrap history pool | 2145689 | 4 inference | Separate node; starts after learner allocation starts |
| k0 learner and current-policy inference | 2145690 | 3 trainer + 3 inference | One six-GPU allocation |

All three jobs explicitly use account `grad-students`, partition `high-priority`, QoS `high-priority`, a 45-day wall limit and no automatic requeue. Candidate GPU nodes are deep-chungus-9, deep-chungus-10 and deep-chungus-11. Other nodes are excluded because Slurm's A100 GRES label alone does not distinguish 40GB from 80GB cards. Startup probes also require healthy A100 devices with at least 79 billion bytes of memory. Node7 is excluded because of its previously faulty device; node8 is down.

The eight-GPU physical node limit requires changing k256's current 1.5B topology from four trainers plus five local inference GPUs and three remote history GPUs to four trainers plus four local inference GPUs and four remote history GPUs. Total k256 allocation remains 12 GPUs. The k0 topology remains three trainers plus three inference GPUs. These changes affect throughput and may affect sampled responses; the loss, question order, batch size and policy-age schedule are unchanged. Every learner stays within one node, so there are no cross-node learner collectives.

CPU-only preparation job **2145682** completed successfully. Model hashes, five native-tokenizer parity probes, dataset validation and preflight passed. All 17,005 questions were included in exactly the same order as the 1.5B study; the longest prompt was 982 tokens. GPU memory fit and full-update execution for 7B are not yet verified. A running Slurm allocation must not be treated as proof that GPU startup or training succeeded.

The submission initially gated GPU jobs on CPU validation job 2145687. The global high-priority job-count limit blocked that CPU job, so configuration and regression checks ran as a one-CPU, zero-GPU step inside existing allocation 2145465, using its node-local Python runtime. All 81 tests passed in 21.59 seconds; 14 upstream deprecation warnings were recorded. Package/control hashes, scientific-setting parity, 32-response batches, 8,192-token context limits, filesystem weight transfer and inference-pool assignments passed validation. The first validation attempt using the shared runtime stalled on NFS reads and only that validation step was cancelled; training processes were untouched.

After validation succeeded, the redundant pending CPU job 2145687 was cancelled. Both learners now depend only on successfully completed asset preparation 2145682; history job 2145689 depends on learner 2145688 starting. `submission.json` records original submissions, while `scheduling-after-validation.json` records the effective dependency changes and verified Slurm fields. At the final queue check, both learners were pending with `QOSGrpJobsLimit` and the history worker was pending on its learner dependency. No 7B update had run. The existing 1.5B jobs 2145464, 2145465 and 2145504 were still running.

Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, pinned revision `916b56a44061fd5cd7d6a8fb632557ed4f724f60`.

Dataset: `zbeeb/Staleness-GRPO-DAPO-Math-17k`, same 17,005-row locked source bytes as the active 1.5B runs.

The prepared release starts from the active 1.5B k0 release, so both k0 and k256 can use independent local inference replicas and filesystem weight transfer. The three application files implementing that existing k0 pool option are preserved under `release/src` here, along with the 7B model constants. The current backup race fix is included. Official imported PrimeRL and vLLM source remain unchanged; no GRPO loss or reward-policy edits were made for this preparation.

Both studies use TensorBoard, node-local runtime/assets/caches and active output, checksum-verified NFS recovery backups, and XFS metric copies. Checkpoints are due at update 1, every 25 updates and the final update. Local retention keeps four recent checkpoints plus 100-update milestones; shared retention keeps two recent verified checkpoints plus milestones. Intermediate evaluation and Runboard remain disabled. Output names are `deepseek7b-dapo17k-6k-exact256-b32-seed42-v1` and `deepseek7b-dapo17k-6k-exact0-b32-seed42-v1`.

For k256, the first 256 learner updates use on-policy cohorts while the remote pool generates historical cohorts from pinned exports. Update 257 begins the exact-age-256 phase. The remote bootstrap pool then finishes; the main inference pool handles later cohorts. The last 256 updates drain the delayed cohort queue. For k0, each update uses the current policy's cohort throughout all 1,000 updates. Neither job is a bounded timing benchmark.

Cluster controls: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek7b-dapo17k-b32-hp-20260929-v1`.
