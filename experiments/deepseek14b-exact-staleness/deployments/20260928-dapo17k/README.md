# Superseded startup attempt: DAPO 17k / DeepSeek 1.5B / exact k256

Jobs 2145416 and 2145429 were stopped during startup validation with zero committed optimizer updates. Native inference startup eventually reached partial rollouts; the delay was not a confirmed deadlock. Its grading journal and logs are preserved. The next release uses independent one-GPU inference servers.

These are the effective submission controls for source commit `abf3a1e` and the new DAPO/6,144-token experiment. They are an audit record, not a command to overwrite or resubmit an existing run.

| Role | Job | Node | HP GPUs | Assignment |
| --- | --- | --- | --- | --- |
| Learner | 2145416 | deep-chungus-5 | 9 × A100 40GB | 4 trainers + 5 local inference |
| Historical worker | 2145429 | deep-chungus-3 | 3 × A100 40GB | Frozen bootstrap-policy rollouts |

Both use account `grad-students`, partition `high-priority`, QoS `high-priority`, and no automatic requeue. The learner requests the full node and all 96 CPUs. Its wall limit is 45 days, which is an allocation limit, not a duration estimate. The remote worker has a 14-day limit and exits once its 256 assigned cohorts are complete.

The initial worker job 2145417 failed before GPU work because the common node-local parent on node 2 was not writable. Retry 2145428 used a private parent but failed the 150-GiB free-disk guard, again before GPU work. Job 2145429 uses a private parent on node 3. Those failures did not change the learner configuration or source release. Their original logs and controls remain on the cluster.

The frozen release is `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek15b-dapo17k-async-hp-20260928-v1/release`. The release binds `MODEL_ID` and `MODEL_REVISION` in `deepseek_study/__init__.py` to the pinned 1.5B model, and installs `manifests/model-deepseek15b.json` as its active `manifests/model.json`. The repository's default model pin remains 14B. All other training adapter source comes from the stated commit; imported PrimeRL remains clean and pinned. `PACKAGE_SHA256.json`, each role's `CONTROL_SHA256.json`, and the run's source snapshot record the actual deployed files.

The release's question manifest was independently generated with the native 1.5B tokenizer and current verifier. It retains all 17,005 questions. Staging checks model/dataset hashes and tokenizer parity again on each node before launching.

Main output name: `deepseek15b-dapo17k-6k-exact256-seed42-v1`. Active data is node-local; verified checkpoints go to the study root's `outputs/` directory on NFS. Journals and TensorBoard events also go to `/mnt/xfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/metrics/` under the same name. Historical export jobs/results are in the NFS study root's `historical-rollouts/` directory under the same name.

Local validation: 598 tests passed, one skipped; focused Ruff and diff checks passed. Cluster startup validated all nine learner GPUs with all-reduce sum 45, BF16 backward, Flash Attention backward and vLLM RMS normalization. Worker probes validate its three GPUs individually; the remote inference pool does not join the learner collective. A running scheduler state does not establish a completed optimizer update; consult `updates.jsonl` and verified checkpoint markers for progress.
