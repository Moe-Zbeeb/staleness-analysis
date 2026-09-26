# Cluster operations

The project root is `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study`. The active organized source is `current`, a symlink to `releases/paper-metrics-20260926`. Work from `current` for the commands below. Its `vendor` symlink reuses the project's isolated `vendor/prime-rl/.venv`; prepared assets are explicitly linked to the verified originals. The old flat source remains preserved outside the active release.

The official checkout and submodules are pinned and unmodified. Caches and managed Python live under the project, and existing model weights and raw data are reused through verified paths. Bootstrap sets `GIT_LFS_SKIP_SMUDGE=1` for dependency checkout: unrelated task assets remain at their recorded LFS pointers. Three initially expanded assets were matched to their recorded content hashes and preserved under the original `diagnostics/cluster/lfs-objects/` before restoring the pointer files. The official source-integrity check remains strict.

## Completed preparation

| Job | Purpose | Result |
| --- | --- | --- |
| 2144915 | Linux installation, asset verification, both configuration profiles, test suite | Completed, exit 0:0; 103 tests passed |
| 2144916 | Initial GPU probe | Failed because the probe used the former `vllm._C` module name |
| 2144920 | Corrected GPU probe and numerical RMSNorm check | Completed, exit 0:0 on all eight A100 80GB GPUs of deep-chungus-9 |
| 2144923 | Launcher source/import checks and prepared-asset checks | Completed, exit 0:0; no training launched |
| 2144954 | First paper-metric dependency installation | Failed before preparation because shell quoting split the dependency argument; corrected in the next job |
| 2144955 | Organized release, pinned Runboard client, both profiles, tests, live metric delivery and XFS mirror | Completed, exit 0:0; 124 tests, three exact backend rows and all 260 synthetic paper scalars verified; CPU only |

The pinned vLLM wheel exposes `vllm._C_stable_libtorch`; the corrected diagnostic loads it and compares the actual CUDA RMSNorm kernel with a Torch reference. NCCL, BF16 backward and Flash Attention backward also passed on every rank. The numerical all-reduce result was 36, derived from the eight-rank world size.

The original 11 model files passed their hashes, the dataset matched its pinned release, and all five native-tokenizer probes passed. The prepared model view is `assets/native-model`; `assets/train-manifest.json` accounts for all 37,713 questions and retains 37,703. No model weights or labels were rewritten.

## Evidence

The small [validation summary](../diagnostics/cluster-validation.json) is committed. Complete logs, submission commands, hardware receipts and deployment hashes are retained under `diagnostics/cluster/` on the cluster and are excluded from Git.

The new release retains its own preparation and metric receipts. Earlier GPU and preflight receipts remain under the original project root. Job `2144955` used no GPUs and loaded no 14B model. Its Runboard project is `staleness-analysis-checks`, and its run metadata explicitly marks the data as synthetic diagnostics. The new relationship charts passed local and hosted-data browser checks. Cloudflare deployment commit `11814e7` succeeded, all hosted assets match the tested Runboard source, and API pagination was verified.

```bash
cat diagnostics/cluster/preparation.json
cat diagnostics/cluster/paper-metrics-validation.json
sacct -j 2144915,2144916,2144920,2144923,2144954,2144955 --format=JobID,State,ExitCode,NodeList
```

## Preparing a future run

Setup and diagnostics never invoke `deepseek-study run`. No study training has been launched. Choose the requested k and a GPU profile before initializing a run configuration. Use the `80gb` profile on a compatible complete eight-A100-80GB node; do not infer every A100 node has the same memory or GPU count.

`deepseek-study init` writes a configuration; `build` resolves official PrimeRL settings; `check` verifies assets. These commands do not start training. `run` starts training and must only be invoked when authorized, inside the matching Slurm allocation.

The current recipe writes full recovery checkpoints every 100 completed optimizer updates and at completion. Under the default root, the shared NFS destination is `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/outputs/<run-name>/checkpoints/step_<N>/`. All 100-step milestones and the final checkpoint are retained. Writes are synchronous; this is shared disk storage, not CPU or GPU memory offload. Intermediate evaluation is disabled, and saved checkpoints can be evaluated separately later. A separate loading/export workflow is needed for evaluation; these sharded recovery checkpoints are not standalone Hugging Face model directories.

Job `2144955` validates the organized source with zero weight decay, the 100-update save interval, paper metrics and the pinned Runboard client. Generate a new run configuration with the requested k; no training configuration or launch is selected automatically by deployment.

The launcher starts separate CPU-only paper and Runboard observers automatically. For [Runboard](runboard.md), use the existing saved connection or the job's `RUNBOARD_SERVER`/`RUNBOARD_TOKEN`, or set `RUNBOARD_DIR` to shared storage. The diagnostic used the saved connection under `/mnt/xfs/home/mohamadzbib/.runboard`. Without a configured destination, dashboard files remain under the run's `tracking/runboard-runs` directory on NFS. No dashboard server, tunnel, backend deployment or training job is started by installation.

SSH inspection found `/mnt/xfs`, not `/mnt/xfs1`. It is the fast NFS4 export; the path's name does not mean node-local XFS storage. The default `metrics_mirror_root` is `/mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics`. Job `2144955` verified 11 copied diagnostic files by checksum there. The mount had about 2.8 TB free globally when inspected. Check capacity before an authorized long run; the complete raw-token archive can approach 96 GiB uncompressed for 1,000 maximum-length updates per copy. Metric archives have no automatic deletion policy. Checkpoints stay on the robust `/mnt/nfs` mount.

For an authorized GPU job, inspect current node topology, then request exactly one full node with `--exclusive`, its full GPU count, and explicit account, partition and QoS. The verified class is `--account=grad-students --partition=high-priority --qos=high-priority`. Constrain placement to nodes with exactly the requested count. When several candidates are eligible, exclude incompatible nodes; Slurm `--nodelist` requests every listed node rather than choosing one candidate. Preserve Slurm's actual `CUDA_VISIBLE_DEVICES` mapping.

Future runs retain immutable source snapshots and require a new output directory. Full 14B end-to-end training, live model-weight transfer, full-context memory fit and actual checkpoint/resume execution still require separately authorized validation. Successful hardware diagnostics do not establish those properties.
