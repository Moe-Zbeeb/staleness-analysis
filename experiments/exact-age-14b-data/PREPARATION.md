# How the three datasets were prepared

Only the final Math-Verify parser-compatible pools are published. “Verifiable” describes reference-format compatibility: it is not independent proof of answer correctness or measured reward accuracy on model outputs.

## Sources and outputs

| Output | Upstream source | Published questions |
|---|---|---:|
| Skywork | `Skywork/Skywork-OR1-RL-Data`, math rows only | 97,809 |
| DeepScaleR | `agentica-org/DeepScaleR-Preview-Dataset` | 37,713 |
| Merged | Independently cleaned Skywork + DeepScaleR pools | 98,941 |

Source revisions are pinned in `configs/sources.lock.json` in the code repository and `source-revisions.json` in each dataset repository. DeepScaleR solutions were never added to the input question. Skywork code tasks are outside these math releases.

## Independent cleaning

Skywork and DeepScaleR were processed separately. Missing prompts or references, malformed text encoding, unsupported prompt structures, and explicit image/diagram markup were excluded. Questions with explicit Answer/Solution section markers were conservatively excluded as suspected answer leakage. Some marked questions may instead contain worked examples; this is a conservative rule.

Original representative question text is preserved. No difficulty, language, tokenizer truncation, generated-response-length, or model-success-rate filter was applied.

## Deduplication inside each dataset

Prompt comparison normalizes Unicode NFKC, whitespace, leading numerical question labels, math delimiters, and selected LaTeX sizing commands. Numbers and mathematical operators remain significant. Normalized-exact copies collapse into one representative.

Near-duplicate candidates use MinHash with 128 permutations, seed 42, and candidate Jaccard threshold 0.8. Confirmation requires at least 0.9 shingle Jaccard, 0.95 character similarity, matching numeric/operator signatures, and at least 80 normalized characters. Shingles contain five tokens. MinHash retrieval is probabilistic and does not guarantee exhaustive semantic deduplication.

Duplicate clusters retain every source-record membership. Ordered reference components must match or pass bidirectional Math-Verify equivalence. An unresolved reference conflict excludes the entire cluster; no answer is chosen arbitrarily.

A question shared across Skywork and DeepScaleR remains in both standalone outputs. Cross-source deduplication never subtracts questions from either standalone dataset.

## Benchmark overlap screening

The screening bank contains 2,006 entries: MATH-500 (500), AIME 2024/2025/2026 (30 each), AMC 2023 (40), AMC 2024 (45), Minerva Math (272), OlympiadBench (674), and Gaokao 2023 English (385). The Gaokao fixture is pinned by Git revision and SHA-256.

Comparison uses lowercase normalized exact matching and conservative near-text matching. Near matches require sufficient shared shingles, then at least one of: Jaccard 0.65, character similarity 0.9, or shingle containment 0.95 with length ratio at least 0.5. Every member prompt in a duplicate group is screened.

A side-by-side audit reviewed 117 borderline pairs. Sixty-five were distinct variants, such as different numbers, expressions, requested quantities, or letter-words. Only those inspected prompt/benchmark pairs are exempted. The public configuration stores normalized prompt hashes and benchmark IDs instead of publishing benchmark question text. Source numbering and equation labels do not automatically veto overlap detection.

Potential overlap is excluded conservatively. This does not establish exhaustive semantic decontamination or absence from model pretraining. Benchmark questions are not included in the published training repositories.

## Merged version

The independently cleaned standalone pools were concatenated, then the same duplicate and reference-conflict rules were applied across the union. Cross-source copies collapse into one representative with both dataset memberships. One unresolved cross-source answer-conflict cluster was omitted from the merge; the standalone pools were unchanged.

Deduplication precedes the final parser-compatibility filter. The publication is the parser-compatible subset of each prepared version, including the separately prepared merged version.

## Final reference filter

Each ordered reference component is parsed with `math-verify==0.8.0`, `LatexExtractionConfig`, string fallback disabled (`fallback_mode='no_fallback'`), and a two-second parsing timeout. Unboxed references are wrapped in `\boxed{...}` for extraction. A row is retained only if every component produces a non-string mathematical parse. Existing reference components and their order are preserved; a list is not automatically interpreted as alternative acceptable answers.

| Version | Clean pool before final parser filter | Parser failures excluded | Published |
|---|---:|---:|---:|
| Skywork | 99,337 | 1,528 | 97,809 |
| DeepScaleR | 37,784 | 71 | 37,713 |
| Merged | 100,472 | 1,531 | 98,941 |

Parser success does not establish mathematical correctness, complete answer semantics, or reliable equivalence grading for arbitrary generated responses. Validate the actual reward implementation on model outputs before training.

## Validation and reproducibility

Preparation tests cover normalization, numeric/operator changes, duplicate collapse, conflict quarantine, independent standalone membership, benchmark matching, reference parsing, and explicit solution markers. The original full pair audit passed all 117 decisions; the public configuration hashes preserve the same decision keys.

Before publication, row counts, reference flags, normalized-prompt uniqueness, prompt structure, and JSONL-to-Parquet round trips are checked. Each release has a Parquet SHA-256 and byte count; uploads are verified against public Hugging Face metadata and file hashes. Only one filtered Parquet pool plus its card and provenance documents is uploaded per repository.

The Hugging Face `train` split is the complete released pool. No held-out validation split was created. For experiments across these overlapping datasets, split question groups consistently across the full union to prevent validation questions from entering another training pool.
