# DeepSeek-R1-Distill-Qwen-7B high-priority studies

These full k256 and k0 studies use the same DAPO17k dataset, 32 responses per update (four questions times eight responses), 1,000 updates, 6,144 response tokens, 2,048 prompt tokens, seed 42 and unchanged GRPO/optimizer settings. Both use the high-priority partition. The k256 jobs use high-priority QoS; k0 uses spec-priority to match the existing node7 baseline, as explicitly requested on September 29. The k256 learner runs on A100 80GB GPUs with a separate bootstrap historical-inference pool.

Slurm reports a per-user high-priority limit of 12 A100 GPUs and a QoS-wide limit of eight running jobs. The separate spec-priority QoS reports an 18-A100-GPU per-user limit and `OverPartQOS`. The k256 pair requests 12 GPUs in high-priority QoS; k0 requests six in spec-priority. Admission still depends on live scheduler limits, other allocations and node availability. Existing 1.5B runs were preserved; no automatic priority fallback is configured.

| Role | Job | GPUs | Placement |
| --- | --- | --- | --- |
| k256 learner and current-policy inference | 2145688 | 4 trainer + 4 inference | One exclusive eight-GPU node |
| k256 bootstrap history pool | 2145689 | 4 inference | Separate node; starts after learner allocation starts |
| k0 learner and current-policy inference | 2145690 | 3 trainer + 3 inference | Node7, spec-priority, after successful completion of 2145504 |

All three jobs explicitly use account `grad-students`, partition `high-priority`, a 45-day wall limit and no automatic requeue. The k256 jobs retain QoS `high-priority` and candidate nodes deep-chungus-9, deep-chungus-10 and deep-chungus-11. The existing k0 job 2145690 was updated in place to QoS `spec-priority`, required node `deep-chungus-7`, and dependency `afterok:2145504`; its previous exclusion list was cleared. No duplicate 7B baseline was submitted. Its unchanged six-GPU, non-exclusive allocation matches the existing baseline's three-trainer/three-inference layout, with 96 CPUs and 384 GiB host memory for 7B.

Node7 previously exposed a faulty eighth GPU. Startup probes require every allocated device to be healthy and to have at least 79 billion bytes of A100 memory; the job fails safely if Slurm supplies a faulty device. The dependency makes the job eligible after the 1.5B baseline completes successfully, including its shutdown and final backup. It does not reserve the node or guarantee an immediate start. A failed predecessor does not trigger the 7B job. `onpolicy-node7-dependency.json` contains the exact update command and verified before/after scheduler fields.

The eight-GPU physical node limit requires changing k256's current 1.5B topology from four trainers plus five local inference GPUs and three remote history GPUs to four trainers plus four local inference GPUs and four remote history GPUs. Total k256 allocation remains 12 GPUs. The k0 topology remains three trainers plus three inference GPUs. These changes affect throughput and may affect sampled responses; the loss, question order, batch size and policy-age schedule are unchanged. Every learner stays within one node, so there are no cross-node learner collectives.

CPU-only preparation job **2145682** completed successfully. Model hashes, five native-tokenizer parity probes, dataset validation and preflight passed. All 17,005 questions were included in exactly the same order as the 1.5B study; the longest prompt was 982 tokens. GPU memory fit and full-update execution for 7B are not yet verified. A running Slurm allocation must not be treated as proof that GPU startup or training succeeded.

The submission initially gated GPU jobs on CPU validation job 2145687. The global high-priority job-count limit blocked that CPU job, so configuration and regression checks ran as a one-CPU, zero-GPU step inside existing allocation 2145465, using its node-local Python runtime. All 81 tests passed in 21.59 seconds; 14 upstream deprecation warnings were recorded. Package/control hashes, scientific-setting parity, 32-response batches, 8,192-token context limits, filesystem weight transfer and inference-pool assignments passed validation. The first validation attempt using the shared runtime stalled on NFS reads and only that validation step was cancelled; training processes were untouched.

After validation succeeded, redundant pending CPU job 2145687 was cancelled. `submission.json` records original submissions and `scheduling-after-validation.json` records the initial validation release. The later node7 scheduling record supersedes those files for k0 placement, QoS and dependency. k256 learner 2145688 remains queued in high-priority QoS; history job 2145689 depends on that learner starting. At the September 29 node7 update, 1.5B learner jobs 2145464 and 2145504 remained running, the 1.5B historical worker had completed, and neither 7B learner had started. Frozen training controls, source hashes and scientific settings were unchanged by the scheduling update.

Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, pinned revision `916b56a44061fd5cd7d6a8fb632557ed4f724f60`.

Dataset: `zbeeb/Staleness-GRPO-DAPO-Math-17k`, same 17,005-row locked source bytes as the active 1.5B runs.

The prepared release starts from the active 1.5B k0 release, so both k0 and k256 can use independent local inference replicas and filesystem weight transfer. The three application files implementing that existing k0 pool option are preserved under `release/src` here, along with the 7B model constants. The current backup race fix is included. Official imported PrimeRL and vLLM source remain unchanged; no GRPO loss or reward-policy edits were made for this preparation.

Both studies use TensorBoard, node-local runtime/assets/caches and active output, checksum-verified NFS recovery backups, and XFS metric copies. Checkpoints are due at update 1, every 25 updates and the final update. Local retention keeps four recent checkpoints plus 100-update milestones; shared retention keeps two recent verified checkpoints plus milestones. Intermediate evaluation and Runboard remain disabled. Output names are `deepseek7b-dapo17k-6k-exact256-b32-seed42-v1` and `deepseek7b-dapo17k-6k-exact0-b32-seed42-v1`.

For k256, the first 256 learner updates use on-policy cohorts while the remote pool generates historical cohorts from pinned exports. Update 257 begins the exact-age-256 phase. The remote bootstrap pool then finishes; the main inference pool handles later cohorts. The last 256 updates drain the delayed cohort queue. For k0, each update uses the current policy's cohort throughout all 1,000 updates. Neither job is a bounded timing benchmark.

Cluster controls: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek7b-dapo17k-b32-hp-20260929-v1`.
