# Cluster operations

The deployed project is `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study`. Its isolated environment is `vendor/prime-rl/.venv`; the official checkout and submodules are pinned and unmodified. Caches and managed Python live under the project, and existing model weights and raw data are reused through verified paths. Bootstrap sets `GIT_LFS_SKIP_SMUDGE=1` for dependency checkout: unrelated task assets remain at their recorded LFS pointers. Three initially expanded assets were matched to their recorded content hashes and preserved under `diagnostics/cluster/lfs-objects/` before restoring the pointer files. The official source-integrity check remains strict.

## Completed preparation

| Job | Purpose | Result |
| --- | --- | --- |
| 2144915 | Linux installation, asset verification, both configuration profiles, test suite | Completed, exit 0:0; 103 tests passed |
| 2144916 | Initial GPU probe | Failed because the probe used the former `vllm._C` module name |
| 2144920 | Corrected GPU probe and numerical RMSNorm check | Completed, exit 0:0 on all eight A100 80GB GPUs of deep-chungus-9 |
| 2144923 | Launcher source/import checks and prepared-asset checks | Completed, exit 0:0; no training launched |

The pinned vLLM wheel exposes `vllm._C_stable_libtorch`; the corrected diagnostic loads it and compares the actual CUDA RMSNorm kernel with a Torch reference. NCCL, BF16 backward and Flash Attention backward also passed on every rank. The numerical all-reduce result was 36, derived from the eight-rank world size.

The original 11 model files passed their hashes, the dataset matched its pinned release, and all five native-tokenizer probes passed. The prepared model view is `assets/native-model`; `assets/train-manifest.json` accounts for all 37,713 questions and retains 37,703. No model weights or labels were rewritten.

## Evidence

The small [validation summary](../diagnostics/cluster-validation.json) is committed. Complete logs, submission commands, hardware receipts and deployment hashes are retained under `diagnostics/cluster/` on the cluster and are excluded from Git.

```bash
cat diagnostics/cluster/preparation.json
cat diagnostics/cluster/gpu-health-2144920.json
sacct -j 2144915,2144916,2144920,2144923 --format=JobID,State,ExitCode,NodeList
```

## Preparing a future run

Setup and diagnostics never invoke `deepseek-study run`. No study training has been launched. Choose the requested k and a GPU profile before initializing a run configuration. Use the `80gb` profile on a compatible complete eight-A100-80GB node; do not infer every A100 node has the same memory or GPU count.

`deepseek-study init` writes a configuration; `build` resolves official PrimeRL settings; `check` verifies assets. These commands do not start training. `run` starts training and must only be invoked when authorized, inside the matching Slurm allocation.

The current recipe writes full recovery checkpoints every 100 completed optimizer updates and at completion. Under the default root, the shared NFS destination is `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/outputs/<run-name>/checkpoints/step_<N>/`. All 100-step milestones and the final checkpoint are retained. Writes are synchronous; this is shared disk storage, not CPU or GPU memory offload. Intermediate evaluation is disabled, and saved checkpoints can be evaluated separately later. A separate loading/export workflow is needed for evaluation; these sharded recovery checkpoints are not standalone Hugging Face model directories.

The deployment receipts above predate the organized source, zero-weight-decay baseline and 100-update save interval. Deploy the current package and regenerate run configs before using this recipe on the cluster.

For an authorized GPU job, inspect current node topology, then request exactly one full node with `--exclusive`, its full GPU count, and explicit account, partition and QoS. The verified class is `--account=grad-students --partition=high-priority --qos=high-priority`. Constrain placement to nodes with exactly the requested count. When several candidates are eligible, exclude incompatible nodes; Slurm `--nodelist` requests every listed node rather than choosing one candidate. Preserve Slurm's actual `CUDA_VISIBLE_DEVICES` mapping.

Future runs retain immutable source snapshots and require a new output directory. Full 14B end-to-end training, live model-weight transfer, full-context memory fit and actual checkpoint/resume execution still require separately authorized validation. Successful hardware diagnostics do not establish those properties.
