# Five-model 1.5B native math sweep

This launcher runs the complete frozen default plan for Qwen2.5-Math-1.5B before GRPO and after step 1,000 at staleness caps 2, 4, 6 and 8: 140 cells, 48,620 responses. It submits no jobs. Root submission must use the saved MadryLab policy: account `grad-students`, partition and QoS `high-priority`, one exclusive full GPU node with its exact live configured GPU count. Submit the GPU job with `afterok` on successful CPU preparation.

## Staging contract

Stage an immutable copy of `evaluation/` and `packages/prime-rl-staleness/src/exact_math.py` under one source root. Put `source-manifest.json` alongside `dispatch.py` with schema:

```json
{"schema_version": 1, "files": {"evaluation/launches/15b-math-20260919/dispatch.py": "SHA256", "evaluation/math_sweep/core.py": "SHA256"}}
```

The example is abbreviated. The manifest must cover the launcher, `evaluation/sweep.py`, all `evaluation/math_sweep/*.py` files, and the three JSON catalogs. Include the frozen grader and other staged sources too. Paths are relative to the staged source root, never absolute. Generate hashes after all source edits; do not modify the package after preparing cells.

Generate the source plan with the staged `sweep.py`, selecting the five exact model IDs listed in `dispatch.py` and `greedy-native sampled-native`. The launcher compares the entire plan with the current frozen catalogs and refuses missing, extra or altered cells. Prepare the full data bundle and its `.receipt.json` through `sweep.py prepare-data`; unresolved overlap candidates still require recorded decisions.

Use the existing validated Python runtime. Configure a persistent Hugging Face cache with enough room for all five pinned models. Initial CPU preparation needs Hub access unless every exact snapshot is already cached. No model training state is read or modified by this launcher.

## CPU preparation

```bash
python evaluation/launches/15b-math-20260919/dispatch.py prepare \
  --plan /absolute/output/plan.json \
  --data /absolute/output/data.json \
  --prepared-root /absolute/output/prepared \
  --results-root /absolute/output/results \
  --tokenizer-root /absolute/production/experiment/root
```

One CPU process prepares cells sequentially and writes `prepared/preparation-index.json` after every completed cell. This index binds the source plan, data bundle and receipt, source package, and each final preparation identity. Restarting preparation validates existing indexed cells and continues missing ones. An exclusive preparation lock prevents concurrent writers. The existing preparer verifies pinned weights and tokenizers each time; preparation can involve substantial repeated reads, so reserve a separate CPU job before requesting GPUs.

## Allocated GPU dispatch

```bash
python evaluation/launches/15b-math-20260919/dispatch.py run \
  --plan /absolute/output/plan.json \
  --data /absolute/output/data.json \
  --prepared-root /absolute/output/prepared \
  --results-root /absolute/output/results \
  --tokenizer-root /absolute/production/experiment/root \
  --workers-from-visible
```

The job must supply `SLURM_JOB_ID`, numeric `SLURM_GPUS_ON_NODE`, and `CUDA_VISIBLE_DEVICES`. The dispatcher compares the allocation count and visible devices with the complete physical `nvidia-smi` inventory. Any mismatch, including exposing eight GPUs on a nine-GPU node, fails. Every physical UUID receives exactly one worker, and every cell subprocess sees only that UUID. Each GPU worker uses four CPU threads; allocate sufficient CPUs for the full node.

Before production, all workers run a private greedy MATH-500 cell containing one question at the normal 3,072-token output cap. Model arms rotate across GPU workers. Every smoke process runs the existing native hardware/context checks, both forced EOS checks, finite decoding probe and one graded generation. Incorrect mathematics is a valid smoke result; a decoding, grading, integrity or runtime failure blocks production. Private smoke profiles, cell IDs and result roots keep these samples out of the 140 production cells.

After all smoke receipts pass, workers draw production cells from one queue, largest maximum output-token workload first. Each cell uses a fresh subprocess, so model/GPU resources are released before the next cell. Completed and partial cells follow the runner's normal receipt validation, locks and resume rules. A hard interruption may leave a torn JSON line; the runner explicitly rejects it instead of silently deleting data.

The job runs offline after CPU preparation. Any worker failure stops all other subprocess groups owned by this dispatcher and leaves durable results for inspection and restart. Signals trigger the same bounded cleanup. No unrelated process or cluster job is signaled.

## Output locations

Production results retain the existing `results/<prepared-cell-id>/` layout for `sweep.py report`. Per-launch state lives separately at:

```text
results/.dispatch/<source-plan-sha-prefix>/<job-id>-<attempt-time>/dispatch.json
results/.dispatch/<source-plan-sha-prefix>/<job-id>-<attempt-time>/logs/gpu-N/<cell-id>.log
results/.dispatch/<source-plan-sha-prefix>/<job-id>-<attempt-time>/smoke-prepared/<private-cell-id>/
results/.dispatch/<source-plan-sha-prefix>/<job-id>-<attempt-time>/smoke-results/<private-cell-id>/
```

`dispatch.json` records allocation UUIDs, source/data/preparation hashes, smoke completion receipts, completed production cells, worker assignments and final status. A shared plan lock prevents another dispatcher from running the same matrix concurrently. Generate final reports using the production prepared/results roots, never the smoke roots.

## CPU verification

```bash
python -m unittest discover -s evaluation/launches/15b-math-20260919 -p test_dispatch.py -v
```

These checks exercise complete GPU mapping, private smoke identity, scope restrictions, the smoke barrier, worker sequencing and subprocess failure propagation. They do not claim allocated-GPU validation; that is the job's mandatory initial smoke phase.
