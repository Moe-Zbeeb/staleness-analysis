# Publication and retention status

Published and verified on 2026-09-23 in the [Hugging Face collection](https://huggingface.co/collections/zbeeb/exact-age-rl-verifiable-math-data-6ab3e8574e1dc838b4df4b99).

| Dataset | Published questions |
|---|---:|
| Skywork math | 97,809 |
| DeepScaleR | 37,713 |
| Merged | 98,941 |

Only parser-compatible references are included. The two standalone datasets remain independently deduplicated; the merged dataset additionally collapses cross-source duplicates.

Public repository file lists, checksums, visibility, and collection membership were verified. Eleven preparation tests and three cleanup tests passed. `reports/publication.json` pins dataset commits and Parquet checksums.

As explicitly requested, the project's raw snapshots, rejected rows, benchmark pools, unfiltered pools, duplicate-format intermediates, and row-level audit files were deleted after verification. Aggregate reports, source manifests, code, and both model snapshots were preserved. Cleanup removed 40 selected paths; retained dataset checksums and model index hashes still match. The detailed deletion receipt is `reports/dataset-cleanup.json`.

The cluster now exposes only these filtered dataset pools at `data/processed/{skywork,deepscaler,merged}/train.parquet`, linked to the immutable release files. To reproduce preprocessing, reacquire the pinned sources. The original preparation report is historical accounting of intermediate pools, not an inventory of retained files.
