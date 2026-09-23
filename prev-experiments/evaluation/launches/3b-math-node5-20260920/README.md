# 3B math evaluation on deep-chungus-5

**Current submission, September 20 at 13:18 UTC:** the user requested using only the free GPUs. Evaluation **2142138** now requests three GPUs on deep-chungus-5 at normal priority, without exclusivity, using this original validated launcher. Full-node job **2142137** is cancelled. Preparation **2142131** continues; report **2142133** now waits for `afterok:2142138`. The same 140 cells and all existing preparation are preserved. GPU startup has not yet occurred. See submission-free-gpus.json. Earlier scheduling entries below are historical.

Superseded scheduling at 13:09 UTC: the user expanded this sweep to all nine GPUs. Original evaluation 2142132 is cancelled; replacement 2142137 uses the [full-node wrapper](../3b-math-node5-full-20260920/README.md). Preparation, cells and outputs remain those recorded here; report 2142133 follows the replacement.

Status at 2026-09-20T12:55:38.342735+00:00: connection restored, all 30 runtime hashes verified, and 10 launcher/device tests passed on cluster Python. Preparation **2142131** is running on deep-chungus-6. Three-GPU evaluation **2142132** is queued for deep-chungus-5 after successful preparation; report **2142133** follows evaluation. All three use normal priority. GPU startup and new generation are not yet verified. Submission and startup receipts are saved alongside this document.

At 12:56:58 UTC, preparation remains running and is downloading the pinned 3B model snapshot. Slurm newly reports deep-chungus-5 as `MIXED+NOT_RESPONDING`; the GPU job still waits on preparation and also requires the node to become available. No node administration or changes to other jobs were performed.

The user requested the three free GPUs on deep-chungus-5 for additional evaluations under the existing normal-priority evaluation authorization. At 12:33 UTC, the node had nine configured GPUs, six allocated to another user, and three free, with 48 CPU slots free. Recheck before submission.

## Matrix

Qwen2.5-3B base and final step-1000 staleness caps 2, 4, 6 and 8. All five immutable model revisions are pinned, including the verified cap-6 release. The same audited data and native greedy/sampled settings as the 1.5B sweep produce 140 cells and 48,620 responses across 12 math benchmarks. AIME 2024–2026 and HMMT February 2026 include 1,024/2,048/3,072 output budgets. These use the 3B model's native 32,768-token context.

Three independent single-GPU workers use Slurm's original CUDA-visible selectors, resolve identities through CUDA and verify physical UUID/PCI identity. All three allocated devices must be idle A100s with identical hardware and at least 85% free VRAM. The launch-specific hardware gate permits 40GB or 80GB A100s; actual hardware is recorded and all five model arms use that same node cohort. Generation settings, questions, seeds, prompts and scoring remain frozen. The GPU stage checks at least 72 GiB available host memory. Five model smoke tests complete across the three workers before production begins.

## Submitted stages

| Stage | Placement | Resources | Dependency |
| --- | --- | --- | --- |
| Preparation | deep-chungus-6 | 4 CPU, 16 GiB, no GPU | None |
| Evaluation | deep-chungus-5 | 3 GPUs, 12 CPU, 72 GiB | Successful preparation |
| Report | deep-chungus-6 | 2 CPU, 8 GiB, no GPU | Successful evaluation |

Every stage explicitly uses grad-students, low-priority partition, normal QoS. The GPU allocation is partial, without exclusivity. Submission records each job ID before releasing held jobs. Source/preparation/result identities and locks prevent duplicate work. The report verifies all 140 cells and answer/decoding receipts before publication.

The local package passed 10 launcher/device tests and nine registry/core tests. Runtime source files are covered by source-manifest.json. This README and status.json are operational notes outside that runtime manifest.

Remote source target: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/evaluation-runs/3b-math-node5-20260920/source`.

Remote output target: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/outputs/math-sweep-3b-node5-20260920`.

Deployment helper and connection evidence: `tmp/eval-3b-node5-20260920` in the workspace. No existing training or 1.5B evaluation jobs were changed.

Latest check at 12:59:44 UTC: node5 is responding again (`IDLE+COMPLETING`), with its previous job completing. Evaluation **2142132** specifically reports `Reason=Dependency`, awaiting successful preparation **2142131**. The first snapshot download finished (8/8 files); preparation is still running.
