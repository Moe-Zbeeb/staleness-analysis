# Runboard integration

The study uses [Runboard](https://github.com/Moe-Zbeeb/runboard) at commit `74b21564d586e43d165d19d2b844ec6cac4deb95` (package version 0.2.0). Bootstrap installs this exact revision into the study environment with `--no-deps`, after installing PrimeRL's frozen environment. Neither dependency's source is modified. Runboard is an additional study dependency; it is not inserted into PrimeRL's lockfile. Its installed version is recorded in the run identity.

## Execution boundary

The launcher starts one CPU-only observer per run, executing `tracking/runboard.py` from the run's source snapshot. The observer reads existing append-only `metrics.jsonl`, `updates.jsonl`, `generations.jsonl`, and checkpoint completion markers. It does not call the model, compute new advantages, change sampling or run evaluation. There is no new callback in the trainer or rollout controller.

PrimeRL's file monitor already writes reduced trainer metrics from rank zero. The observer forwards only rows whose producer is `trainer`, plus selected study metrics. One Runboard run combines these sources. Network requests, retry and spooling run in the observer and Runboard's sender thread. Observer startup or process failure does not fail training; `logs/runboard.log` contains its diagnostics. Monitoring still consumes CPU, memory and storage bandwidth, so include that overhead in throughput comparisons.

On normal completion or a handled failure, the launcher stops the training services, records `run-status.json`, and gives the observer up to 20 seconds to drain and exit. The observer waits for this status even if the controller has already written `study-complete.json`, so final trainer metrics can arrive first. If the launcher disappears, the child detects that and exits as crashed. Abrupt termination can interrupt delivery; the original study journals remain available for a later import.

## Destination and credentials

The observer uses `RUNBOARD_DIR` when set. Otherwise, it uses the endpoint and token in `RUNBOARD_SERVER` / `RUNBOARD_TOKEN` or Runboard's saved `server.json` under `RUNBOARD_HOME` (default `~/.runboard`). If a complete connection is unavailable, it writes local Runboard files to `<output_dir>/tracking/runboard-runs`, on NFS under the default study root.

Configure an existing personal backend on the cluster with the installed client:

```bash
vendor/prime-rl/.venv/bin/runboard configure https://runboard.example.workers.dev
```

Replace the example URL with your backend. The command prompts for the token; do not put credentials in study JSON or Git. A saved connection on a laptop does not establish that compute nodes have the same configuration or network access. A shared cluster home can provide the configuration to its compute nodes. This integration does not deploy a Cloudflare backend or copy credentials between machines.

For a shared local dashboard directory, set `RUNBOARD_DIR` to your chosen NFS directory before launch. To view it, run `runboard serve --dir /path/on/nfs/runboard-runs` on an authorized host and use the authenticated URL it prints. Starting a dashboard server is separate from training and is not automatic.

The default dashboard project is `staleness-analysis`; `RUNBOARD_PROJECT` overrides it. `DEEPSEEK_STUDY_RUNBOARD=0` disables the automatic observer. These are tracking settings, not scientific recipe fields.

Runboard uploads numeric metrics and run metadata, including its standard hostname, command-line and Slurm identity. The observer does not upload response text, question IDs, response IDs, model weights, checkpoint files, or credentials. Source/config fingerprints and scientific settings are recorded as metadata. Runboard connection details are not written to the study's config or tracking receipt.

## Metric meanings

| Dashboard series | Source and horizontal axis | Interpretation |
| --- | --- | --- |
| `staleness/*` | Update receipt; completed optimizer update | Learner and behavior versions, exact minimum/maximum age, bootstrap flag |
| `rollout/consumed_*` | Update receipt; completed optimizer update | Reward, truncation, zero-advantage fraction and response count of the consumed behavior cohort; these are not held-out or current-policy evaluation |
| `queue/*` | Update receipt; completed optimizer update | Pending cohort count and serialized training payload bytes |
| `generation/total_*` | Update receipt; completed optimizer update | Cumulative generated cohorts, responses and output tokens |
| `generation/<purpose>/*` | Generation receipt; behavior policy version | Generation time, output tokens and intended consumption step; purposes are bootstrap (`warmup`), `on_policy`, and `deferred` |
| `trainer/*` | Official trainer file monitor; its recorded optimizer update | Existing loss, ratio/clipping/mismatch summaries, entropy, optimizer, performance, disk and timing metrics |
| `checkpoint/*` | Complete-bundle marker; saved optimizer update | Completed checkpoint event and version; checkpoints themselves remain on NFS |

The trainer metrics retain their upstream meanings. `trainer/loss/mean` is the mean of normalized microbatch losses, not the complete update objective. `trainer/optim/lr` is logged after the scheduler advances and describes the next update. `trainer/perf/throughput` uses packed input slots, not only generated/loss tokens. `trainer/source_time_unix` retains the source timestamp. Dashboard arrival time differs from source time, especially during retrospective imports; use the step axis for scientific comparisons. Generation-version charts use a different clock from consumption-step charts, as the table states.

Partial JSONL lines wait for completion. Malformed complete lines produce a warning and are skipped; rewriting or truncating an already-read file stops the observer to avoid replaying it silently. Only completed checkpoint markers create checkpoint events. Metric values remain scalar; the existing sample-to-response ordering limitation is not bypassed by uploading positional joins.

## Offline delivery and importing saved logs

Runboard retries temporary HTTP failures and spools unsent data under `$RUNBOARD_HOME/spool` (normally `~/.runboard/spool`). Its `runboard sync` command uploads that spool later using your configured backend. This is different from local file mode, which writes dashboard-ready storage directly.

To import a completed run's original journals, or reconstruct a dashboard after observer failure:

```bash
vendor/prime-rl/.venv/bin/deepseek-study track /absolute/path/to/run --once
```

`--once` requires a study-completion marker or launcher termination status. Omit it to follow an existing run live. Each manual invocation creates a new dashboard run, leaving previous imports intact; repeated imports are separate entries. The automatically launched observer uses the launcher's unique run ID. Each resumed training segment has its own output directory and dashboard run, carries its starting step and resumed flag, and uses absolute optimizer-update indices. No automatic cross-segment metric merge is performed.

Receipts are saved as `<output_dir>/tracking/runboard-<id>.json` with the project, run ID, mode and final observer status. They contain no token. A `finished` receipt reports observer completion, not independent confirmation of remote delivery; inspect the backend and any spool files for that.

## Validation and deployment status

Local tests exercise the actual pinned Runboard SDK in file mode and through an authenticated local HTTP server, checking the exact run and row count. They cover partial records, metric clocks, metadata, terminal states, network-outage spooling, the real observer subprocess, final trainer-log draining, and launcher tolerance of observer failure. No model is loaded and no training is launched by these checks.

CPU-only Slurm job `2144955` deployed the pinned client and passed all 124 study tests. Its synthetic diagnostic verified three exact backend rows, including all 260 paper scalars, under project `staleness-analysis-checks`; it also verified 11 files in the XFS mirror. The run is explicitly marked `diagnostic_only`, not a training result. The relationship-chart interface passed both local and hosted-data browser checks. Cloudflare deployed [deployment commit 11814e7](https://github.com/Moe-Zbeeb/runboard-cloudflare/commit/11814e7); every deployed asset matches the tested source, and API pagination was verified. Later real-model metric-delivery and recovery checks are recorded in the [bounded readiness test](readiness-test.md). Intermediate evaluation stays disabled, checkpoints remain every 100 updates plus completion on NFS, and weight decay remains zero.

## Paper metrics and relationship charts

The [paper guide](paper-metrics.md) inventories the BAPO/M2PO measurements. In addition to existing logs, the observer reads `paper-metrics.jsonl` and `evaluation-metrics.jsonl`. They appear under `paper/` and `evaluation/`. The separate paper observer combines complete rank token archives, persists derived data and mirrors it to the configured XFS root. It runs even if Runboard delivery is disabled. The live Runboard observer waits for the paper observer's final status before closing.

The updated Runboard dashboard understands `config.runboard_binned_charts`. It displays latest-step mean/quantile relationships with explicit source steps. Individual bin histories remain available through the metric filter. Full raw scatter/word-cloud data remain on NFS and XFS; they are not uploaded as response text or token arrays. Updating an independently hosted dashboard requires deploying its new static assets.

## Direct GRPO contribution

Use `paper/gradient_signal/noncontributing_token_fraction` for the global fraction of response tokens with no direct GRPO loss gradient. It includes zero advantages and clipping that stops the gradient, with separate cause fractions. Tokens merely outside the ratio bounds can still contribute. The trainer records the masks in each raw shard, and the observer weights by token counts across all ranks. These series are automatically plotted and mirrored to XFS with the other paper metrics. See the [precise definition and numeric edge cases](paper-metrics.md#tokens-with-no-direct-grpo-contribution).

CPU-only job `2144956` validated this addition against the live backend: 75% surrogate-clipped tokens and 25% zero-advantage tokens gave a noncontributing fraction of 1.0. All 281 synthetic scalar values and the XFS mirror matched; all 128 tests passed on Linux. No Runboard library or dashboard change was necessary for these additional scalar charts.
