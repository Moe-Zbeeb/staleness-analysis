# DeepSeek 1.5B exact256 resilient relaunch

Production job **2145376** was submitted on September 28 with high-priority partition and QoS, account `grad-students`, 12 A100 GPUs across `deep-chungus-5` and `deep-chungus-7`, 48 CPUs and 200 GiB per node, and a 45-day time limit. Four GPUs train on node5; two generate on node5 and six generate on node7. Hardware and startup verification results are recorded in the launch controls. A RUNNING scheduler state alone is not proof of completed startup.

The model remains `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, with 1,000 total updates and exact lag256 after 256 bootstrap updates. All study hyperparameters and generation settings match job 2145304. This is a clean restart because the previous run failed after 15 committed updates without a full recovery checkpoint. Its outputs remain intact.

## Changes and evidence

The failure response contained `2^{2019} - \left(1 + e^{-\frac{1}{e}}\right)^{2019}` against reference `-1`. `math-verify` repeatedly timed out expanding/comparing it. The new policy retries prediction verification at 8 then 32 seconds per symbolic operation. Persistent prediction timeouts receive reward zero with an explicit unverified status; reference or infrastructure failures still stop the run. CPU validation job 2145374 replayed the exact saved token sequence through the isolated worker in 42.75 seconds, returning the expected timeout label without crashing. No response was regenerated or dropped.

The new manifest was rebuilt from all 37,713 source questions. Its 37,696 included IDs and 17 exclusions are identical to the previous manifest. The grader identity changed, so this run and future comparison baselines must use the new policy. TensorBoard logs the fraction of consumed responses whose verification timed out; detailed attempts and extracted answers remain in `grading.jsonl`.

Recovery checkpoints are scheduled at update 1, every 5 updates, and the final update. The custom trainer adapter and controller share one schedule. Checkpoints include trainer/optimizer/scheduler, per-rank RNG, sampler and delayed-cohort queue state. Local retention keeps four recent saves and every 100-update milestone; verified shared backups keep two recent saves and those milestones. Backup retention never republishes intentionally pruned saves on each poll. The first live production checkpoint still needs verification once update 1 completes.

401 local tests passed, one optional test skipped. Tests cover timeout recovery, fatal reference timeouts, the production large-exponent answer, checkpoint schedule agreement, RNG/queue restore, incomplete/corrupted checkpoints, backup publication and retention. Official PrimeRL and vLLM source remain unmodified. Multi-node automatic restart remains disabled; an explicit recovery must validate deployment compatibility and the complete checkpoint before loading.

## Locations

Cluster root: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study`.

- Controls: `launches/deepseek15b-k256-12gpu-hp-20260928-v7`.
- NFS backups: `outputs/deepseek15b-exact256-12gpu-seed42-v7`.
- Local working files: `/tmp/staleness-storage-v2/mohamadzbib/deepseek15b-exact256-12gpu-seed42-v7`.
- XFS metrics: `/mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics/deepseek15b-exact256-12gpu-seed42-v7`.
- TensorBoard events: `tensorboard/` beneath the run and each metric backup.

The frozen release, prepared-data identity, control checksums, exact submission arguments and validation receipt are retained in the controls. See the repository diagnostic `diagnostics/resilience-2145376.json` for the submission snapshot.

Startup verification: all 12 allocated GPUs passed BF16, FlashAttention and per-node six-rank collectives (sum 21). The twelve-rank cross-node collective returned 78 and broadcast 160 MiB in 0.4441 seconds. Slurm remains RUNNING. Inference engines are initializing; the first committed training update and checkpoint are not yet verified. This broadcast diagnostic is not a training throughput measurement.
