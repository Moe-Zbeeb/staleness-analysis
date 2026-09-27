# TensorBoard logging

New launches use TensorBoard only. The runtime never starts Runboard, even if old Runboard environment variables are inherited. Official PrimeRL remains pinned and unmodified: its file monitor writes JSONL and a separate CPU observer exports those existing records. This changes observation, not the GRPO objective, token masks, queue, optimizer or policy-version clock.

## Files and viewing

Each run writes events under `<run-directory>/tensorboard/` on NFS. `tracking/tensorboard.json` records ownership, status, clocks and errors. Events are flushed and copied incrementally to `<metrics_mirror_root>/<run-name>/tensorboard/` on XFS. The final `tensorboard-mirror-manifest.json` contains event-file checksums. Raw journals and paper artifacts continue to use the separate metric mirror; checkpoints remain on NFS.

Use TensorBoard 2.20.0 from the project environment, with access to either shared event directory:

```bash
vendor/prime-rl/.venv/bin/tensorboard --logdir /absolute/path/to/run/tensorboard --host 127.0.0.1 --port 6006
```

Open `http://127.0.0.1:6006` on the machine running the server, or use an authenticated SSH tunnel for a remote server. The training job writes events; it does not expose an unauthenticated public dashboard. The event directory can also be copied to a local machine for viewing.

To export or repair events for a terminated run from its saved journals:

```bash
vendor/prime-rl/.venv/bin/deepseek-study track /absolute/path/to/run --once
```

Existing events are verified against the source records, and matching scalar events are not written again. Changed historical values or append-only journals fail explicitly. Use the original source environment for historical runs whose source identity differs.

## Metric clocks

| Series | Step axis |
| --- | --- |
| `trainer/*`, controller/staleness/reward/queue metrics, `paper/*`, checkpoints | Optimizer update number |
| `generation/{warmup,on_policy,deferred}/*` | Behavior policy version used to generate the cohort |
| `inference/*` and other native records with no optimizer step | Unix time in milliseconds; wall time is also saved |
| `evaluation/*` | Saved checkpoint update number from offline import |

Inference telemetry is not assigned a fabricated optimizer update. Use TensorBoard's wall-time axis to view it by time. Generation may run ahead of consumption; its reward is distinct from `rollout/consumed_reward_mean`. Bootstrap updates are explicitly labeled, and update 257 is the first exact-age-256 update for the current full study.

All finite numeric trainer and paper diagnostics are exported. This includes `paper/gradient_signal/noncontributing_token_fraction`, its zero-advantage/clipped/numerical components, GRPO loss, ratios, KL estimators, entropy, reward, lengths, throughput, bin histories, queue sizes and checkpoint milestones. The noncontributing fraction is logging only: no token or update is removed. Raw token arrays remain separate artifacts; TensorBoard scalars do not replace those arrays or constitute held-out evaluation results.

## Shutdown and failure

The launcher stops training services, records terminal status, drains paper calculations, then lets TensorBoard flush the final events. An observer failure is reported in its log/status and does not modify training. Journals remain available for replay. Training progress must still be checked through committed update receipts and Slurm state; a dashboard process or event file alone does not prove training is healthy.

Runboard remains an optional dependency only for explicitly inspecting historical artifacts. Its old observer and documentation are retained; it is absent from the default installation and new launch path.
