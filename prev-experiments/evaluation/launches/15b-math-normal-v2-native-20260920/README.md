# Resume the remaining 1.5B evaluations on chungus-11

The user requested continuing the 1.5B base and staleness ≤2/4/6/8 benchmark sweep on deep-chungus-11. The prior normal-priority, two-free-GPU authorization remains in effect.

## Submitted continuation

Worker **2142130** started at **12:26:49 UTC on September 20, 2026**, on **two A100 80GB GPUs of deep-chungus-11**, account `grad-students`, partition `low-priority`, QoS `normal`, eight CPUs and 24 GiB requested host memory. It has a 24-hour limit and no dependency. Other users’ jobs and all training jobs are untouched.

Completed node1 worker **2142048** has **60 complete cells and 15,175 saved responses**. Their receipt identities are recorded in [the submission receipt](submission.json). The continuation processes only node11's original **80 cells  / 33,445 expected responses**, preserving the full 140-cell/48,620-response matrix. Existing preparations, partial outputs, model revisions, seeds, prompts, token limits, graders and comparison groups are reused.

The existing CPU report **2142054** was held, its failed dependency replaced with `afterok:2142130`, and released. No duplicate report job was submitted. The unchanged report code verifies completeness and identity for all 140 cells before producing the combined report, including node1’s 60 preserved results.

## GPU-selection correction

The old mapper interpreted Slurm CUDA selectors as Linux device minor numbers. This wrapper captures Slurm's original `CUDA_VISIBLE_DEVICES` before environment activation, preserves it with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, obtains identities only from the visible CUDA devices, and validates their UUIDs and PCI identities against NVIDIA's physical inventory. Linux minor numbers are used only to verify device files after identity resolution. Every pre-existing compute process is still rejected, and the frozen 85% free-VRAM threshold and A10080GB cohort remain enforced.

The frozen dispatcher and scorer are not modified. Only its allocation function is replaced at runtime, with separate hash-bound provenance stored in `results/.native-recovery/JOB-RESTART/`. The original launcher identity continues to bind preparation and comparison cells; the added allocation wrapper is explicitly recorded in each dispatch allocation.

## Verification at 12:27 UTC

The allocation and host-memory checks passed. Slurm assigned selectors **6,7**, which resolve to UUIDs `GPU-669ea827-23e4-39f5-eed4-5feebc0c837c` and `GPU-8700e8c0-2ab7-4c31-3e0a-bcfc35597de8`, physical Linux minors 5 and4. Both GPUs had 81,153 MiB free and no compute processes. Available host RAM was approximately 846.6 GiB. These values confirm why the old same-numbered-minor mapping was incorrect.

Eleven local regression tests passed, along with Python compilation, shell syntax and deployed-source hash checks. Tests cover the historical mapping failure, noncontiguous selectors, duplicate GPUs, PCI mismatch, changed CUDA visibility, occupied devices, cohort identity and preserved frozen source. At **12:28:58 UTC**, the dispatcher had verified/reused the 80 derived cells and started one model smoke-test subprocess on each GPU. Dispatch status was `smoke`, with no error. Those startup tests and new production cells were not yet complete.

## Evidence and paths

- [Submission and effective scheduling fields](submission.json)
- [Allocation, memory and GPU identities](allocation-2142130.json)
- [Startup snapshot](startup-2142130.json)
- [Source manifest](source-manifest.json)

Source manifest SHA256: `475910d50fd6cd392f19688044c347dd3b66da018fd63391bb2eca915f091aad`.

Remote wrapper: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/evaluation-runs/15b-math-normal-v2-native-20260920/source`.

Existing output: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/outputs/math-sweep-15b-normal-v2-20260919`.

Inspect worker 2142130, report 2142054 and existing result receipts before any future retry. Do not resubmit the finished node1 shard or regenerate completed cells.
