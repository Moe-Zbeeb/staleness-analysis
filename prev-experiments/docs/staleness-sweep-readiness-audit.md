# Four-model staleness sweep readiness audit

Audited September 16, 2026. Target: Qwen2.5-Math-1.5B, Qwen2.5-3B, Qwen2.5-Math-7B, and Qwen3-14B at staleness caps 4, 6, and 8, compared with the existing cap-2 experiments.

**Verdict: the pinned training runtime supports the complete 12-run sweep, but the repository is not yet ready to launch it.** All 12 candidate configurations pass PrimeRL's typed parser with 1,000 trainer and orchestrator updates and no unexpected setting differences from their frozen references. However, these are audit-time candidates: the 12 experiment directories, launch files, and positive-cap validation gates have not been implemented. The existing cap-0 tooling cannot be submitted unchanged for this study.

This audit covered configuration resolution, baseline provenance, loss and reward code, data/tokenizer preservation, rollout freshness and weight synchronization, GPU mapping, launch/recovery scripts, checkpoint completion, evaluation attribution, and telemetry. No production source was changed, no SSH session was opened, and no cluster job was submitted during this audit. CPU checks establish compatibility and specific invariants, not GPU integration or successful full training.

## Findings requiring action

### 1. Missing sweep arms and an incompatible validation gate — P1

There are four baseline experiment directories and four prepared cap-0 directories. There are no cap-4, cap-6, or cap-8 experiment directories. A different TOML cap by itself does not provide an isolated run, recovery state, checkpoint link, telemetry identity, or validated launch.

The reusable guard explicitly sets the expected cap to zero in `experiments/staleness-zero/scripts/staleness_guard.py:45`, requires resolved cap zero at line 100, requires every age metric to equal zero at line 112, and requires `policy.start == policy.end == step - 1` at line 121. Its gate at line 134 also requires exactly five smoke updates. These checks would reject legitimate positive-cap training.

Generate each arm independently from its frozen cap-2 reference. Parameterize the guard by the declared cap K. Validate raw effective training traces against `0 <= (step - 1) - policy.start <= K` and `policy.start <= policy.end <= step - 1`, using the shipped cohort's update step. Preserve finite loss/gradient and paired-checkpoint checks. Use a 25-update smoke with production batch size and token limits; five updates cannot expose an age-8 boundary. Update every copied helper and its source fingerprint together.

### 2. Legacy small-model smoke scripts disagree with production topology — P1

| Model | Frozen production allocation | Legacy smoke health probe | Legacy device mapping |
| --- | --- | --- | --- |
| 1.5B | 3 trainer + 2 inference = 5 GPUs | 8 processes | Assumes 6+2 on eight devices |
| 3B | 4 trainer + 3 inference = 7 GPUs | 9 processes | Assumes 6+3 on nine devices |

Locations: `experiments/dapo-qwen25-math15b-grpo/scripts/smoke_job.sh:24,42` and `experiments/dapo-qwen25-3b-grpo/scripts/smoke_job.sh:24,42`. Under the matching five/seven-GPU allocation, the health probe requests too many local GPU processes. Copying these legacy scripts would prevent a reliable validation run.

The prepared cap-0 copies already use the correct process counts and derive device ordering from the frozen topology. Reuse that corrected implementation when generating the sweep, and verify every generated smoke and production launcher against its configuration.

### 3. Hardware placement and the 7B control need explicit matching — experiment comparability

The final 1.5B and 3B baselines started from scratch on their 3+2 and 4+3 profiles. Their retained logs identify NVIDIA A100-PCIE-40GB devices. Preserve those profiles and GPU class. Do not change trainer or inference GPU counts to fit available nodes while describing staleness as the only treatment difference.

The 7B baseline used 4+2 until checkpoint 975, then resumed on 8+1. The audit's 7B candidate uses the frozen final 8+1 configuration; it is not an exact hardware match to all 1,000 historical updates. A clean fixed-topology comparison needs the same fixed profile for caps 2, 4, 6, and 8, including a fresh cap-2 control. Otherwise retain the historical control with this confound explicitly documented. That experimental choice remains unresolved.

The current shared submitter requires a healthy full A100 node with exactly the configured GPU count (`experiments/staleness-zero/scripts/submit.py:21-31,49`). The saved inventory had no matching five-GPU node and its seven-GPU node was down. This is historical inventory, not a statement about current availability. Matching the original small-model partial allocations requires an intentional placement policy; resizing requires newly matched cap-2 controls.

The submitter also matches generic `gpu:a100` rather than device memory/subtype. The health probe prints device details but does not enforce the baseline class. Record and check the actual class at startup, alongside GPU count and parallelism.

### 4. Static parity does not enforce complete resolved-runtime parity — P2

The current guard checks frozen TOML equality after allowed cap/path substitutions, but its runtime audit reads only the orchestrator cap. An unexpected CLI override or installed dependency default could change training settings without violating that check. See `experiments/staleness-zero/scripts/staleness_guard.py:33-58,97-100`.

Before accepting a run, compare normalized resolved trainer, orchestrator, and inference settings against the frozen reference. Allow only the declared cap, run identity/path relocation, node-local model staging, verifier cache paths, and a validated same-arm resume step. Capture the installed runtime/package provenance as well as the pinned PrimeRL commit. A clean PrimeRL checkout alone does not fingerprint its installed dependencies.

The audit compared locally resolved references with retained production resolved JSON. Differences were operational model/cache locations, the recorded 7B resume step, and inference wrapper fields absent from the server's own dump. No additional learning-setting differences were found. This manual audit is evidence; it is not yet an automatic launch guard.

### 5. Duplicate jobs can write the same arm — P2

The shared submitter records the queue but does not reject an already active job for the same experiment (`experiments/staleness-zero/scripts/submit.py:52-63`). The training wrapper accepts an existing matching manifest and may resume or relocate that run's output without an exclusive run lock; for example, `experiments/dapo-qwen25-3b-grpo-stale0/scripts/train_job.sh:119-125,158-184`.

Separate directories protect different arms. They do not prevent two submissions of one arm from racing over checkpoints, recovery state, or observer files. Add duplicate submission detection and an atomic per-arm runtime lock that has a defined crash/requeue recovery path.

### 6. Training completion is checked more strongly than evaluation completion — P2

The reusable guard requires all requested training updates, finite metrics, effective training traces, and a paired final trainer/orchestrator checkpoint. It does not establish that every final benchmark completed with its expected number of examples and final-policy provenance (`experiments/staleness-zero/scripts/staleness_guard.py:97-128`).

In the pinned orchestrator, `wait_for_version` warns and continues after timeout (`src/prime_rl/orchestrator/orchestrator.py:479-502`). Therefore successful update-1000 artifacts alone are insufficient to declare the entire research run complete. Require final benchmark coverage/counts, no missing or failed evaluations, and final `policy.start == policy.end == 1000`, alongside the existing training checks. Keep training-finished and full-study-complete states distinct so recovery can finish evaluation/validation without silently accepting missing results.

## Configuration parity that passed

PrimeRL is pinned to `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1` (v0.9.0). The audit parsed the frozen production TOML for each model using its actual `RLConfig` and CLI parser, then changed only `orchestrator.max_off_policy_steps` to 4, 6, or 8. All 12 parsed successfully and had zero unexpected differences from the corresponding locally resolved cap-2 reference.

| Model | New caps | Trainer + inference GPUs | Inference TP / DP | Trainer / orchestrator updates | Special settings preserved |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-Math-1.5B | 4, 6, 8 | 3 + 2 | 1 / 2 | 1,000 / 1,000 | Original model/tokenizer and data |
| Qwen2.5-3B | 4, 6, 8 | 4 + 3 | 1 / 3 | 1,000 / 1,000 | Original model/tokenizer and data |
| Qwen2.5-Math-7B | 4, 6, 8 | 8 + 1 | 1 / 1 | 1,000 / 1,000 | Final baseline profile; historical topology caveat above |
| Qwen3-14B | 4, 6, 8 | 7 + 2 | 2 / 1 | 1,000 / 1,000 | Non-thinking tokenizer, optimizer CPU offload, resharding |

The configurations preserve batch size 64, group size 8, learning rate 1e-6, AdamW weight decay 0.01, gradient norm limit 1, 30 warmup updates, PPO clipping at 0.2, seed settings, and initial/minimum/maximum concurrency 64/16/128. The custom objective remains group-mean-centered reward advantages without standard-deviation normalization, a reference KL term, or a critic.

Training preserves temperature 1, a 3,072-token completion limit, and 4,096-token context. The same 17,005 ordered prepared DAPO rows and exact terminal-answer verifier must be copied and hash-verified. The 14B prepared data differs in tokenizer-derived `prompt_tokens` metadata; the data comparison excludes that field and checks the remaining row content and order. Copy the already corrected 14B tokenizer/data rather than regenerating them.

Evaluation settings remain unchanged: regular MATH500/AIME24/AIME25/AMC23 evaluation every 100 updates with greedy decoding and the existing 3,072-token limit; Minerva/Olympiad at updates 0 and 1,000 with their 2,048-token limit; sampled AIME24/25/26 at updates 0 and 1,000 with eight samples per question, temperature 0.6, and a 3,072-token limit. These per-benchmark differences are part of the existing setup and should remain fixed across caps.

Each production arm must start from the original base model with fresh optimizer, scheduler, task-source, and observer state. Never initialize a higher-cap treatment from trained cap-2 weights. Resume only that arm's own verified checkpoint. Existing recovery checks support paired checkpoints and protect run fingerprints; their deliberate handling of interrupted final checkpoints should be retained when generalizing the tooling.

## Runtime semantics and measurement limits

The upstream schema accepts all nonnegative caps and defaults to 8 (`packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:527`). The train sink checks queued samples against `(step - 1) - cap` before shipment and on insertion. The dispatcher cancels stale live-policy training groups before weight synchronization. Samples exactly at the cap remain eligible. No upstream loss or synchronization patch is needed for this sweep.

Keep `TARGET_LAG = 1` unchanged. It is a separate admission/batch lead constraint, not a cap of one on old queued or unfinished rollouts. Cap 8 permits age 8; it does not force age 8. Measure realized age distributions to establish how different the treatments actually were.

Surviving unfinished generations retain their prefix and cached state across weight synchronization: the vLLM pause uses `mode="keep", clear_cache=False`. Preserve this cap-2 behavior for positive-cap arms. Applying the discarded cap-0 idea of restarting every unfinished generation would add a second treatment change.

`policy.start` is stamped when a group opens, including its later-dispatched siblings; `policy.end` is stamped when an episode is handled. These are conservative group/episode provenance fields, not per-token weight-version records. Raw trace checks remain necessary because the aggregate age helper clamps negative values.

Intermediate evaluation uses the shared live inference pool and can overlap subsequent weight updates. Its logged single policy version is the minimum starting version, which does not prove a frozen checkpoint evaluation. Preserve the baseline scheduling for this sweep and avoid attributing every intermediate score to one exact checkpoint without inspecting all provenance. Strict checkpoint evaluation should use a separately frozen policy consistently across arms.

Under normal completion, final evaluation is triggered after version 1,000 is applied and no further training update follows. All 123 retained final-evaluation examples sampled from the three completed small-model baselines had policy span `[1000, 1000]`. That is sample evidence, not a complete final-evaluation audit. The retained 14B metrics snapshot is too old to establish its current completion status.

`off_policy/dropped` counts discarded queued traces, not all cancelled in-flight work. Dispatcher cancellation counters do not enter the existing file monitor, and the Opik filter omits `time/wait_for_policy` even though the raw metrics contain it. Do not interpret absent counters as zero or assume changing the Opik filter can reconstruct missing file data. Preserve training behavior and apply any additional observation consistently to all newly matched arms.

## Verification completed

Evidence is retained in the workspace directory `tmp/cap-sweep-audit-20260916/`, next to this repository.

| Check | Result | Evidence |
| --- | --- | --- |
| Actual typed parser and baseline comparison | 12/12 candidates passed; 1,000 updates in both components | `resolved-parity.json` |
| Retained production resolved settings vs local reference | Only explained operational/serialization differences | `runtime-reference-diffs.json` |
| Pinned freshness, queue insertion, cancellation, and decomposition methods | 40 focused cap/step cases passed | `check_runtime_staleness.py`, `runtime_staleness_results.json` |
| Actual custom PPO loss, clipping, gradients, masking, empty mask | 20 CPU cases passed | `loss-cpu-checks.json` |
| Actual terminal-answer verifier | 24 existing regression cases passed | `reward-cpu-checks.json` |
| Existing zero-cap guard regression suite | 14 tests passed; does not validate positive-cap arms | `experiments/staleness-zero/scripts/test_staleness_guard.py` |
| Python/shell/TOML/JSON syntax across eight experiment trees | 163 files passed | `syntax-and-source-checks.json` |
| Prepared cap-0 source fingerprints | 96 file hashes verified | `syntax-and-source-checks.json` |

Runtime unit evidence executes unchanged method bodies extracted from the pinned source with mocked surrounding state. Loss/reward checks use the local CPU dependencies. These tests do not exercise CUDA, distributed collectives, vLLM serving, live cluster dependency versions, or long-running recovery. Exact inspected upstream source and hashes are retained under `pinned_runtime/`; additional source analysis is in `runtime_findings.md`.

## Acceptance before full training

1. Resolve the 7B control and small-model placement while preserving the chosen per-model topology across caps.
2. Generate 12 isolated experiment trees from verified baseline references, with only the declared cap and operational identities changed. Freeze and verify source, model/tokenizer, data, and environment provenance.
3. Generalize the guard, derive launcher GPU mappings consistently, enforce resolved-runtime parity, and protect each run from duplicate writers.
4. Pass a 25-update GPU smoke for every model/cap with production generation settings, raw provenance bounds, finite learning metrics, and paired checkpoint evidence. Inspect realized age distributions; reaching K is not a correctness requirement.
5. Start each production run fresh and train to update 1,000, resuming only its own checkpoints if interrupted. Confirm every update and the final checkpoint pair, then separately verify complete final evaluation before declaring the arm finished.

The complete experimental design and prior cap-2 age evidence are in [staleness-cap-sweep.md](staleness-cap-sweep.md).
