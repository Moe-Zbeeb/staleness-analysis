# 7B staleness ≤4 relocation to deep-chungus-9

At **2026-09-19 23:19:11 UTC**, existing job **2142052**, restart **1**, was relocated from `deep-chungus-10` to `deep-chungus-9` and released. At **23:20:39 UTC**, Slurm reported **RUNNING** on node9 and the current attempt had passed isolated-runtime verification. **GPU health, checkpoint restoration and resumed training steps remain unverified.**

The user authorized running 7B ≤4 elsewhere after the recommendation to use five available GPUs on node9. This supersedes the earlier node10 placement restriction recorded in the frozen recovery specification.

| Setting | Recorded value |
| --- | --- |
| Job / restart | 2142052 / 1 |
| Requested node | deep-chungus-9 |
| Account / partition / QoS | grad-students / low-priority / normal |
| GPU layout | Four trainer GPUs + one inference GPU |
| CPU / host memory request | 48 CPUs / 384 GiB |
| Validated paired resume checkpoint | 225 |
| Scheduling sequence | Hold existing job; update node request; release existing job |

The relocation created no replacement training job. Training settings, recovery source, checkpoint paths, observer state, and frozen manifests remain unchanged. The package manifest SHA-256 remains `0bd3e9f6d022ff2ab1a0c0126deedb57388434f2da28eb28633dec59a438ef5b`.

Both nodes use the same reviewed GRES device-file rule under SHA-256 `1a5c2624ff52be8400be632e1aed6f61b75399446b91d9eb249d19ac38307732`. Startup must derive a fresh allocation mapping on node9, verify five idle A100 80GB-class GPUs, pass the BF16/NCCL check, restore the paired checkpoint, and demonstrate successful training and weight updates. Earlier node10 health output cannot establish these checks for restart1.

## Evidence

- [Local relocation receipt](relocation.json), including scheduler state before, while held, and after release.
- [Current-attempt startup observation](startup-20260919T232039Z.json), binding job2142052/restart1 to node9 and the unchanged isolated runtime.
- [Subsequent connection failure](connection-check-failure.txt). The next read-only SSH connection closed before returning a report; remote checks stopped. This does not establish that the Slurm job stopped.
- [Read-only restart1/node9 monitor](../../../../../tmp/relocate-7b4-20260920/postlaunch_status.py), which excludes prior-attempt batch output.
- [Frozen recovery description](../../README.md).
- Remote durable receipt: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/recoveries/7b-stale4-weight-sync-20260919/relocations/job-2142052-node9-20260920/relocation.json`.

This document records the placement change separately from the original frozen specification. Subsequent startup evidence should retain its own observation time and restart identity.
