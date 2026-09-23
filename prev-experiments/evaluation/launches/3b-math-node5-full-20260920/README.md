# 3B evaluation using all nine GPUs on deep-chungus-5

**Superseded at 13:18 UTC:** the user requested only the free GPUs. Full-node job2142137 is cancelled; replacement2142138 uses the original three-GPU launcher at normal priority. Existing preparation and report are preserved. Do not resubmit this full-node wrapper without a new user instruction.

Submitted at 2026-09-20T13:09:56.412500+00:00: full-node evaluation **2142137** requests all nine GPUs at normal priority and waits for preparation **2142131**. The superseded three-GPU job **2142132** is cancelled. Existing report **2142133** now depends on successful completion of **2142137**. GPU startup is not yet verified.

The user expanded the existing normal-priority evaluation from three GPUs to the complete node on September 20. At 13:02:55 UTC, Slurm reported nine A100 GPUs, zero allocated GPUs, 96 CPU slots, 500,000 MiB configured host memory, an idle state and no remaining jobs on deep-chungus-5.

This wrapper preserves the original 140 prepared cells, 48,620 responses, model revisions, dataset bundle, prompts, generation settings, scoring implementation, preparation job 2142131 and report 2142133. It validates the frozen source manifest `a0b2a53a0c73ff78f3257a1f730dccc327217275570ebc0cd72aff8086fb836c` before consuming its existing preparation index. It does not rewrite that source snapshot.

The full-node job requests `grad-students`, `low-priority`, `normal`, one exclusive node, exactly nine A100 GPUs, 36 CPUs per task and 216 GiB RAM. Slurm's original CUDA assignment is preserved. All nine GPU identities are checked through CUDA against their physical UUID/PCI identity, and each allocated GPU must be idle and have sufficient VRAM. The host must have at least 216 GiB available memory. Nine separate UUID-bound GPU probes must pass BF16 backward before the original dispatcher runs. Five model smoke tests precede production; the production pool uses all nine workers, one GPU per independent cell.

Submission holds the old pending evaluation 2142132 and report, submits and validates the replacement while held, updates the report dependency, cancels only the superseded evaluation, and releases the replacement and report. Preparation remains running throughout. Submission and startup receipts record the resulting job IDs and actual runtime status.

The local tests cover all nine GPU identity mappings, verification of all nine GPU probes before dispatch, and rejection of any probe failure. Runtime files are covered by source-manifest.json. This document and operational receipts are outside that runtime manifest.

Remote wrapper: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/evaluation-runs/3b-math-node5-full-20260920/source`.

Existing frozen source: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/evaluation-runs/3b-math-node5-20260920/source`.

Existing prepared inputs and results: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/outputs/math-sweep-3b-node5-20260920`.

Latest status at 13:11:23 UTC: preparation **2142131** has prepared **28/140 cells**, all 28 base-model cells, and is fetching the next model. Full-node evaluation **2142137** remains pending the preparation dependency. Another user has since started job **2142136** on six node5 GPUs, so only three GPUs are currently free and the exclusive nine-GPU evaluation also needs the node to become fully available.
