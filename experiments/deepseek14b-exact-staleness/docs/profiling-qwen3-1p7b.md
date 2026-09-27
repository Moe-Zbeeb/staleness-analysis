# Qwen3-1.7B timing profile on node 4

This profile uses the requested [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B), pinned to revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. It keeps the supplied tokenizer and chat template with native thinking mode enabled by default. It does not insert a DeepSeek thinking prefix or add a reasoning parser. The full generated completion remains available to the existing grader and learner.

The profile preserves the frozen 14B training configuration: cleaned DeepScaleR, 64 prompts × 8 responses per update, 2,048 prompt tokens, 8,192 response tokens, seed 42, temperature 1.0, the same optimizer and GRPO loss, zero weight decay, no intermediate evaluation, lag 256 and the configured 1,000 updates. Sampling stays at the study settings rather than adopting the model card's recommended temperature/top-p/top-k. A separate supervisor stops after four committed bootstrap updates and the final learner timing record, with a five-hour run bound and six-hour Slurm limit. Four bootstrap updates do not measure exact-256 steady-state behavior or checkpoint costs.

The requested node `deep-chungus-4` exposes nine A100 GPUs. Use all nine: four trainers and five data-parallel inference workers, tensor parallelism one. The exclusive job requests nine A100 GPUs, 96 CPU slots and all node memory under account `grad-students`, partition/QoS `background`, with no requeue. The existing high-priority 14B allocation leaves insufficient high-priority quota for this node; background is authorized and matches the other timing profiles. Model family, tokenizer, response lengths, GPU memory and inference-worker count can affect timing. This is not a model-size-only comparison.

Before downloading the model, every allocated device must pass an independent BF16 forward/backward probe with unique GPU identities. The nine-rank collective probe must sum to 45 and pass BF16 backward, Flash Attention backward and vLLM RMSNorm on every rank. A failed device stops the job before training. Device assignments are derived from Slurm's actual visible-device order and recorded in `allocation.json`.

Official PrimeRL and the shared environment remain unchanged. The generated isolated adapter changes the model pin and the model-specific tokenizer parity checks, just as the [Qwen2.5 profile](profiling-qwen3b.md) does. Prepared token IDs, decoding, native rendered prompts and BOS/EOS are compared against the pinned Qwen tokenizer. Model assets are hashed, retained question IDs and order must equal the 14B baseline, and source/configuration differences are recorded in `preparation.json`. The model loader already supports Qwen3 and tied embeddings.

Run names use `qwen3-1p7b-exact256-timing-<job-id>`. Metrics go to Runboard and the XFS mirror, while original logs and rollouts stay on NFS. `work/timing-summary.json` records bounded completion versus failure or timeout. Existing 14B, 1.5B and 3B jobs are not modified.

## Submitted job and startup evidence

Job **2145018** started on `deep-chungus-4`, using source commit `7dc3786`, background partition/QoS, all nine A100 PCIe **40GB** GPUs, 96 CPU slots and all node memory. Inference uses devices 0–4 and training uses devices 5–8. All nine devices passed independent BF16 checks and the collective/Flash Attention/vLLM probes. The collective sum was 45. Native tokenizer parity passed all five probes; preparation retained the same 37,703 questions in the same order as the 14B baseline.

At **2026-09-27 00:48:11 UTC**, the four-rank learner had initialized and was waiting for startup weight broadcast; the five-worker inference service was starting. Runboard had registered the run. No optimizer update had completed at that observation, so complete-cohort execution and update timings remained unverified. This is a timestamped launch record, not a live status report.

A prior submission, **2145017**, was canceled before allocation or training because the submission guard incorrectly rejected Slurm's equivalent pending node-count notation `1-1`. The guard was corrected to accept both `1` and `1-1`, then the same command was resubmitted. No earlier training output was reused. The [launch receipt](../diagnostics/profiling-20260927-qwen3-1p7b.json) records both attempts, resource verification, hardware, preparation and run identity.

Cluster paths:

- Control directory: `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/profiling/20260927-qwen3-1p7b-node4`.
- Isolated release and configuration: `work/release` and `work/study.json` below that directory.
- Run output: `work/qwen3-1p7b-exact256-timing-2145018`.
- Bounded timing result: `work/timing-summary.json`, written when the profile exits.
- XFS mirror: `/mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics/qwen3-1p7b-exact256-timing-2145018`.
- [Runboard](https://runboard-cloudflare.mbz02.workers.dev): project `staleness-analysis`, name `qwen3-1p7b-exact256-timing-2145018`, run ID `f2ee3b9b7ae64ec09c960ded2bbfd112`.

Local validation passed six source-integrity and bounded-supervisor tests, Ruff and shell syntax checks. Runtime validation above is recorded separately from those local tests.
