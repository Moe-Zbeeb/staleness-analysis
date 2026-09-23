# Exact-age RL math data

Three public releases contain only parser-compatible references:

| Version | Questions | Hugging Face |
|---|---:|---|
| Skywork math, independently deduplicated | 97,809 | [Dataset](https://huggingface.co/datasets/zbeeb/Skywork-OR1-Math-Verifiable-Dedup) |
| DeepScaleR, independently deduplicated | 37,713 | [Dataset](https://huggingface.co/datasets/zbeeb/DeepScaleR-Verifiable-Dedup) |
| Merged, cross-source deduplicated | 98,941 | [Dataset](https://huggingface.co/datasets/zbeeb/Skywork-DeepScaleR-Merged-Verifiable-Dedup) |

Shared questions remain in both standalone datasets. Only the merged version collapses cross-source duplicates. “Verifiable” means reference parsing with Math-Verify; it does not establish independently proven answer correctness.

Read [PREPARATION.md](PREPARATION.md) for the complete method and per-version accounting. Dataset payloads live on Hugging Face. This directory contains the preparation and publication code, pinned sources, reviewed-match hashes, and aggregate reports. No raw, rejected, benchmark, or unfiltered question pools are committed.

## Reproduce

Use Python 3.12 and an isolated environment. Substantial preparation should run in an appropriate batch allocation.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-prep.txt
mkdir -p reports manifests
.venv/bin/python scripts/acquire.py --root "$PWD" --kind dataset
.venv/bin/python scripts/fetch_gaokao.py --root "$PWD"
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/prepare.py --root "$PWD"
.venv/bin/python scripts/export_release.py --root "$PWD" --output release --code-revision "$(git rev-parse HEAD)"
```

The preparation pass creates intermediate pools for auditing. Publication reads only `reference_parseable.jsonl` from each version. `export_release.py` enforces release counts and produces one Parquet training pool per dataset. Parser timeouts and library/platform differences can affect a fresh run; a count mismatch stops export for inspection.

To publish the validated exports with an authorized Hugging Face token:

```bash
.venv/bin/python scripts/publish_release.py --folder release
```

The release configuration fixes the account, repository names, visibility, and new collection title. The publisher checks the account, restricts the file list, records commits, verifies public file metadata/hashes, and confirms collection membership. Credentials are read through Hugging Face's standard authentication mechanism, never embedded in source.

`refine_benchmark_screen.py` and `finalize_screen.py` preserve the one-time audit correction code used for the initial preparation. The regular rebuild entry point is `prepare.py`. For portability, public acquisition omits the original machine-specific weight-reuse path, and benchmark exception keys are stored as hashes; the mathematical cleaning rules are unchanged.
