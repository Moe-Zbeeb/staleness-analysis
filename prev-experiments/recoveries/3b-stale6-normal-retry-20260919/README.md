# 3B staleness <=6 normal-priority retry

This package continues the existing run from the complete step-775 checkpoint after job 2142027 failed because the orchestrator waited 1200 seconds for the trainer startup policy.

The requested placement is three GPUs on `deep-chungus-10` at normal priority, alongside the separate 7B job. Two GPUs train and one GPU serves inference. The original experiment, data manifest, checkpoint location, Comet project, model, optimizer, scheduler and staleness bound remain unchanged.

The only additional resolved configuration change from the previous three-GPU recovery is `orchestrator.ckpt.wait_for_weights_timeout`: `None` (effective default 1200 seconds) to `7200`. Inference TP1 and the earlier three-GPU topology overlay remain unchanged. Full checkpoint state is read directly from its original shared path; there is no checkpoint staging or output-directory relocation.

Run `validate_job.sh` in a CPU allocation before submitting `train-stale6.sh` in a three-GPU normal-priority allocation on `deep-chungus-10`. The preflight verifies all six differences from the frozen four-GPU config, checkpoint step 775, orchestrator next step 776, two trainer shards, optimizer/scheduler metadata, unchanged data/source fingerprints, and device mapping. The launcher captures immutable resume evidence before Prime rewrites runtime configs.

`package-manifest.json` covers the immutable package and initial evidence. Tests, bytecode, later validation output and runtime attempt captures are excluded. No scheduling, deployment or submission is performed by this package build.
