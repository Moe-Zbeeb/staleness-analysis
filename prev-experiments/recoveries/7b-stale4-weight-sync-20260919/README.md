# Qwen2.5-Math-7B cap-4 weight-sync recovery

This package recovers the original five-GPU run after job 2142033 failed during inference weight synchronization. The recorded valid paired checkpoint is step 225. Step 239 contains only orchestrator progress and must be preserved as an orphan, never used as a training resume point.

## Bounded runtime change

The job uses its own source tree, `prime-rl-source`, copied from Prime-RL commit `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1`. Across 238 upstream source/configuration files, only `src/prime_rl/orchestrator/clients.py` changes: its existing `update_weights` definition is replaced by an import of `weight_sync_recovery.update_weights`. The helper preserves the call signature and performs one pause, one acknowledgement, one update and one resume, with a 3,600-second HTTP read timeout. An uncertain or failed update propagates the failure and does not resume inference or retry the collective.

Original and patched source manifests, the pristine client source and `runtime-provenance.json` bind the change. Before launching, `runtime_guard.py` checks every original source/configuration file against the live shared repository, checks the job-local tree, verifies exact patch reconstruction, and verifies that the launcher, configs, inference server, client and replacement function resolve to the isolated source. The original shared repository's commit and clean-tree checks remain in the launcher. No installed package, shared repository or other job is patched.

`PYTHONPATH` is set only in this job to the recovery root, isolated source/config packages and frozen experiment Python files. The component launchers inherit that environment. CPU preflight uses the same path and import checks.

## Unchanged experiment and resume state

The original experiment, data/tokenizer manifests, run directory, model revision, cap 4, optimizer, scheduler, sampling, reward and evaluation definitions remain frozen. The topology stays four trainer GPUs and one TP1/DP1 inference GPU. The existing startup-only overlay remains `orchestrator.ckpt.wait_for_weights_timeout = 7200`; the runtime fix adds no resolved configuration differences.

Preflight verifies checkpoint 225's metadata and orchestrator progress hashes, four trainer shards, trainer progress 225, orchestrator next step 226, and model/optimizer/scheduler/progress state coverage. Training refuses a fresh restart. The launcher chooses a complete paired checkpoint, and capture verifies all four shards. Before Prime-RL can overwrite runtime configs, the guard copies paired checkpoint metadata/progress and resolved configs into immutable per-attempt evidence, bound to this runtime patch.

Incomplete checkpoints are moved into the original run's recovery tree under the current job/restart identity. The launcher records the destination and saves an SHA256 of any orphan orchestrator progress before moving it. Thus the orchestrator-only step 239 is retained, while paired step 225 supplies the actual resume state. Existing attempt logs and metrics remain in the original run. Prime-RL's normal same-arm resume cleanup still handles future rollout/broadcast scratch state.

## GPU and process ownership

The job uses the existing reviewed Slurm GRES → device minor → UUID → numeric CUDA mapping and the same five-rank BF16/NCCL health probe. Before CUDA health checks or model initialization, UUID-targeted NVIDIA queries require A100 80GB-class devices, at least 85% free memory and no existing compute processes on any selected GPU. A busy allocated GPU fails the job without selecting a different device or signaling its processes. Only this launch's own training and observer processes are managed.

Submission requires fresh authentication, current node/job/free-GPU inspection, paired checkpoint validation, and verification that no existing run lock is held. The intended placement remains normal priority on five GPUs of deep-chungus-10. This package does not submit a job.

## Provenance and validation

The package manifest pins the source tree, helper, launcher, guards, checkpoint evidence and tests. `launcher-provenance.json` links the prior recovery and runtime manifest. Per-job runtime import proofs are written under `validation/`; checkpoint capture and final audit include the runtime provenance and source-manifest hashes. The retention-aware final audit preserves the prior captured-checkpoint rule, so pruning step 225 after enough newer complete checkpoints does not invalidate an otherwise proven resume.

Local checks exercise delayed/failed HTTP updates, the original duplicate-update behavior, the patched receiver/watcher sequence, import-source boundaries, complete pristine upstream file verification, allocated-GPU occupancy, and checkpoint retention evidence. These CPU checks do not establish GPU weight transfer success. Before calling the recovery healthy, verify restoration from 225 and several consecutive successful weight updates under the allocated runtime.
