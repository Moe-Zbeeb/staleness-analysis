# Normal-priority evaluation on two shared nodes

The user authorized two available GPUs on each of `deep-chungus-1` and `deep-chungus-11` at normal priority. This package runs the five-model, 140-cell Qwen2.5-Math-1.5B native suite on those partial allocations. It performs no Slurm submissions and never changes the original source package, completed CPU preparations, model snapshots or training jobs.

## Immutable preparation and hardware cohorts

The original high-priority preparation remains the source of all 140 cells. This launcher verifies its original code manifest, plan and preparation index, then derives new cells into a separate prepared root. Only these top-level fields change:

- `hardware`: A100 minimum 37 GiB on node 1, or 75 GiB on node 11.
- `comparison_sha256`: binds the original comparison, hardware cohort and assigned node/shard.
- `transition`: records the original cell/preparation, this launcher, assigned node and derivation purpose.
- `cell_id`: the digest of the resulting cell.

Model weights and revisions, input-file hashes, runtime versions, tokenizer, prompts, gold answers, decoding settings, context, seed and sample counts remain identical. Prompt JSONL bytes are copied exactly. Derived preparations have separate receipts and one `preparation-index-shard-N.json` per shard. Both node cohorts receive derived identities, including node 11 whose minimum memory remains 75 GiB.

Each complete `(profile, benchmark)` group, including every token budget and all five models, stays on one fixed node. The deterministic partition balances maximum output-token work:

| Shard | Node | GPU cohort | Cells | Responses | Maximum output tokens |
|---|---|---|---:|---:|---:|
| 0 | deep-chungus-1 | A100 40GB | 60 | 24,080 | 57,098,240 |
| 1 | deep-chungus-11 | A100 80GB | 80 | 24,540 | 57,185,280 |

This keeps base-versus-trained and token-budget comparisons within the same node and hardware cohort. Across-benchmark throughput is not a hardware-controlled comparison.

## GPU ownership

The GPU mapper verifies a pinned live `gres.conf`, resolves each allocated `SLURM_JOB_GPUS` ordinal through the configured device-file list, checks the device-file minor, and reads its UUID from `/proc/driver/nvidia/gpus`. It does not interpret numeric `CUDA_VISIBLE_DEVICES` as NVIDIA enumeration indices.

Exactly two distinct allocated UUIDs become two workers. UUID-targeted NVIDIA queries verify the expected A100 40GB/80GB class and enough free memory for the frozen 85% vLLM reservation. Original visibility, Slurm IDs, GRES records, device minors, PCI identities, UUIDs, memory and process evidence are saved in the launch receipt. No unrelated GPU is selected or signaled.

Each node tests all five model arms in private one-question greedy smoke cells, sequentially across its two GPU workers. The original runner verifies native context, EOS handling, finite decoding and grading before production. Smoke cells use separate roots and profiles and cannot fill production counts. Every production cell runs the immutable original `sweep.py` in a fresh UUID-scoped subprocess.

## Invocation

The original CPU preparation must be complete. Generate `source-manifest.json` for this package after all changes, with `schema_version: 1` and `files` mapping package-relative filenames to SHA256 values. Include all runtime Python and shell files, plus the documentation and tests if desired. The original staged package keeps its original manifest.

```bash
python /new/package/newdispatcher.py run \
  --source-root /immutable/original/source \
  --plan /original/output/plan.json \
  --data /original/output/data.json \
  --prepared-root /original/output/prepared \
  --derived-prepared-root /new/output/prepared \
  --results-root /new/output/results \
  --original-results-root /original/output/results \
  --tokenizer-root /production/project/root \
  --workers-from-visible --shard 0 --shards 2 \
  --gres-config /usr/local/etc/gres.conf \
  --expected-gres-sha256 REVIEWED_SHA256
```

Use shard 1 on node 11. If omitted, `--original-results-root` defaults to the original prepared root's sibling `results` directory. A shared lock there excludes the original whole-matrix launcher; separate exclusive shard locks prevent duplicate partial jobs. The saved partition prevents accidental reassignment or mixing different sources. Each worker uses four CPU threads.

Production results remain compatible with the original report command. `report.py` additionally requires the union of both derived indexes to contain exactly all 140 original cells once before generating the combined report. Submit the report job only after both GPU jobs succeed.

## Output and recovery

Launch state, allocation evidence, private smoke sources/results and per-cell logs are under `results/.partial-dispatch/<plan-hash-prefix>/<job-id>-shard-N-<attempt>/`. `dispatch.json` records the source, launcher, partition, derived index, actual hardware, smoke outcomes and production receipts.

Failures and signals stop only process groups created by this dispatcher. Existing production receipts and partial records are retained. Restarting the same shard verifies the same partition and derivations, performs fresh smoke checks, and delegates completed/partial-cell handling to the original runner. Torn JSON lines require explicit repair.

## CPU checks

```bash
python -m unittest discover -s evaluation/launches/15b-math-normal-20260919 -p 'test_*.py' -v
```

These checks cover GRES/minor/UUID mapping, actual hardware bounds, free memory, disjoint group-preserving shards, immutable derivation, comparison identities, all-five-model smoke checks and the original full-matrix lock. Actual GPU validation happens at job startup.
