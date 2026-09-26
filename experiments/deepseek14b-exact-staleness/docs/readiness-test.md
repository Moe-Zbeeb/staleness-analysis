# Bounded 14B cluster readiness test

This diagnostic uses one exclusive node with eight A100 80 GB PCIe GPUs under the `grad-students` account, `high-priority` partition and `high-priority` QoS. Four GPUs run the trainer and four run inference. It does not start the prepared production exact-256 run or benchmark evaluation.

## Result

Job **2144960 passed**, exit `0:0`, on `deep-chungus-11` in **54m 45s**. The tested runtime source is commit `377e5887f9e4582d8a0c797365cca9c1e9b285c3`. All 129 cluster tests passed; local validation passed 128 tests with the CUDA-only case skipped. Lint passed.

| Check | Observed result |
| --- | --- |
| Hardware | All eight A100 80 GB GPUs passed NCCL, BF16 backward, Flash Attention backward and vLLM RMSNorm checks |
| Weight transfer | Startup and subsequent policy broadcasts passed, including restored policy version 2 |
| Initial run | Three updates with exact ages 0, 1, 1; 96 unique responses and 531,134 response tokens |
| Length and memory | Responses reached 8,192 tokens in every cohort; rank-zero peak reserved trainer memory was 69.68 GiB initially and 72.00 GiB after restart |
| Recovery | Checkpoint 2 resumed into a new directory and completed update 3 with the original 32 queued responses and 153,675 tokens |
| Queued-data identity | Response IDs, payload digest, policy versions, token IDs, positions, behavior log probabilities and advantages matched |
| Forward replay | All saved current token log probabilities matched exactly across all four ranks |
| Hosted Runboard | 26 initial and 8 resumed rows fetched with pagination; all 1,504 paper scalar values across the four update events matched the archives exactly |
| XFS mirror | All 44 initial and 24 resumed mirrored files verified by checksum |
| Production | Exact-256 output remains absent; no production training or benchmark evaluation launched |

The replay is **not bitwise identical after backward**. Its gradient norm was `0.06119707599282265`, versus `0.061200711876153946` originally, although the saved forward log probabilities and logged loss matched. A targeted checkpoint check of the first layer's input-normalization weight and Adam state found maximum absolute differences of `1.19e-7` in weights, `2.79e-8` in the first moment and `1.23e-14` in the second moment; the optimizer step matched exactly. This was a four-tensor spot check, not a comparison of every model tensor. Strict deterministic GPU training is not configured, and the diagnostic does not promise bitwise optimizer trajectories. No numerical or algorithm setting was changed to hide these differences.

Each diagnostic recovery checkpoint is about **177.3 GB (165.1 GiB)**. NFS saves took roughly 4–5 minutes, and the restart's loading stage took about 14 minutes. These are observations from this node and shared-storage load, not production throughput estimates. Use `paper/tokens/count` and study response counts for scientific token budgets; upstream trainer throughput and progress counters have the different meanings documented in the [Runboard guide](runboard.md).

## Scope

`scripts/readiness.py` first runs the contract suite, an in-memory simulation of the complete requested exact-256 schedule, eight-rank CUDA/NCCL/Flash Attention checks, and a separate trainer-to-inference NCCL transport probe. The synthetic schedule verifies 256 bootstrap updates followed by 744 updates with exact age 256, with no unused cohorts.

The real-model diagnostic then uses these explicit overrides:

| Setting | Production | Diagnostic |
| --- | --- | --- |
| Lag | 256 | 1 |
| Updates | 1,000 | 3 |
| Prompts × responses | 64 × 8 | 4 × 8 |
| LR warm-up | 30 updates | 1 update |
| Checkpoint interval | 100 updates | 2 updates, plus completion |
| Output | Prepared production directory | Separate diagnostic directory |

The model, prepared data, 2,048-token prompt cap, 8,192-token response cap, four/four GPU split, mixed precision, optimizer settings and GRPO objective are unchanged. Lag 1 makes actual delayed-cohort consumption testable within three updates. A real exact-256 consumed cohort cannot appear before update 257.

After the three-update run, the harness resumes checkpoint 2 into a separate output directory and reexecutes update 3. It requires the original queued response IDs, payload digest and learner/behavior versions to match. This checks recovery of existing queued data; it does not establish bitwise reproduction of future inference.

The harness also audits update counts and exact ages, raw token evidence, paper metrics, noncontributing-token fractions, completed checkpoint components and the XFS metric mirror. Runboard uses the separate `staleness-analysis-readiness` project.

## Issues found and corrected

| Job | Finding | Resolution |
| --- | --- | --- |
| `2144957` | The 14B model loaded, but the trainer-to-inference NCCL communicator failed before any optimizer update. | Initialize PrimeRL's topology-based transport choice in the parent launcher before any child process creates NCCL groups. |
| `2144958` | A small eight-GPU probe reproduced the late-initialization failure; both early socket and early peer/shared-memory configuration passed. | Keep the isolated probe as a regression stage before loading model weights. |
| `2144959` | Weight transfer and generation passed: 32 graded responses, 186,397 response tokens. The token exporter then mixed CPU rollout fields with GPU outputs after backward and before the first optimizer update. | Move detached contribution-mask inputs to the model-output device; test CPU rollout metadata with GPU outputs against actual GRPO gradients. |
| `2144960` | All acceptance stages passed after both fixes. | Deploy the identical tested runtime source and retain the original attempt logs. |

Both fixes are in the study package. Official PrimeRL and its submodules remain pinned and unmodified. See the [integration inventory](upstream-integration.md).

## Evidence and reproduction

Each attempt has a separate source release, submission receipt, Slurm log and diagnostic output under `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study`. Full logs and model checkpoints are not committed to Git.

The successful evidence directory is `diagnostics/cluster/readiness-2144960` beneath that root. It contains `readiness.json`, the independent `verification.json` and its `verify.py`, phase logs, initial and resumed output directories, and recovery checkpoints. A compact copy of the results is committed in [cluster-validation.json](../diagnostics/cluster-validation.json). Runboard project `staleness-analysis-readiness` contains initial run `76faf1e625784c2bacb53be9e3d127aa` and resumed run `d6eb479ef296472190a2af7c4949ff6e`.

Within an authorized, exclusive eight-A100-80GB Slurm allocation, run from the deployed release:

```bash
vendor/prime-rl/.venv/bin/python scripts/readiness.py --study configs/exact256-80gb-seed42.json --directory /absolute/path/to/a/new/diagnostic-directory
```

This command executes a bounded real training diagnostic. It is not a preparation-only command. The directory must be new, and the prepared production output must remain absent. The harness bounds the initial model run to one hour and resume to 40 minutes; the Slurm allocation has a two-hour limit.

Passing the small test does not validate the production 512-response batch, a 256-cohort queue's peak host memory and checkpoint size, long-run stability, or sustained production throughput. These remain distinct from component and recovery readiness.
