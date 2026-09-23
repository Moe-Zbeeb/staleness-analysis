import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from prepare import normalize, prompt_leakage_marker


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code-revision', required=True)
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parents[1]
    spec = json.loads((code_root / 'configs/release.json').read_text())
    source_lock = json.loads((code_root / 'configs/sources.lock.json').read_text())
    report = json.loads((args.root / 'reports/preparation-report.json').read_text())
    summary = {}
    for name, entry in spec['datasets'].items():
        source = args.root / 'data/processed' / name / 'reference_parseable.jsonl'
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        assert len(rows) == entry['expected_rows']
        assert all(r['answers'] and r['verifiability']['all_reference_components_parse'] for r in rows)
        assert all(not prompt_leakage_marker(r['prompt']) for r in rows)
        assert all(r['messages'] == [{'role': 'user', 'content': r['prompt']}] for r in rows)
        normalized = [normalize(r['prompt']) for r in rows]
        assert len(set(normalized)) == len(rows)
        if name != 'merged':
            assert all(r['datasets'] == [name] for r in rows)
        folder = args.output / name
        (folder / 'data').mkdir(parents=True, exist_ok=True)
        table_rows = [{**r, 'reference_answer_raw': json.dumps(r['reference_answer_raw'], ensure_ascii=False), 'source_records': json.dumps(r['source_records'], ensure_ascii=False)} for r in rows]
        parquet = folder / 'data/train.parquet'
        pq.write_table(pa.Table.from_pylist(table_rows), parquet, compression='zstd')
        check = pq.read_table(parquet).to_pylist()
        assert check == table_rows
        source_names = ['skywork', 'deepscaler'] if name == 'merged' else [name]
        manifest = {'dataset': name, 'repo_id': entry['repo_id'], 'rows': len(rows), 'code_repository': spec['github_repository'], 'code_revision': args.code_revision, 'code_directory': spec['github_directory'], 'reference_parser': 'math-verify==0.8.0', 'string_fallback': False, 'label_correctness_independently_verified': False, 'input_sha256': digest(source), 'train_parquet_sha256': digest(parquet), 'train_parquet_bytes': parquet.stat().st_size, 'training_split_is_unsplit_pool': True, 'preparation_counts': report['versions'][name]}
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (folder / 'source-revisions.json').write_text(json.dumps({'training_sources': [s for s in source_lock if s['key'] in source_names], 'benchmark_sources': [s for s in source_lock if s['repo_type'] == 'dataset' and s['key'] not in ['skywork', 'deepscaler']], 'gaokao': json.loads((code_root / 'configs/gaokao-source.json').read_text())}, indent=2) + '\n')
        title = {'skywork': 'Skywork OR1 Math — Verifiable, Deduplicated', 'deepscaler': 'DeepScaleR — Verifiable, Deduplicated', 'merged': 'Skywork + DeepScaleR — Verifiable, Cross-Deduplicated'}[name]
        method = {'skywork': 'Skywork was processed independently: retain its math rows, clean them, deduplicate within Skywork, screen benchmark overlap and prompt leakage, then retain parser-compatible references. Shared questions with DeepScaleR remain in this dataset.', 'deepscaler': 'DeepScaleR was processed independently: clean its questions, deduplicate within DeepScaleR, screen benchmark overlap and prompt leakage, then retain parser-compatible references. Shared questions with Skywork remain in this dataset.', 'merged': 'The two independently cleaned pools were merged, cross-source duplicate questions were collapsed, unresolved cross-source answer conflicts were quarantined, and only parser-compatible references were retained. The standalone datasets were not modified by merging.'}[name]
        license_header = 'license: mit\n' if name == 'deepscaler' else ''
        sources_yaml = ''.join('  - ' + s['repo_id'] + '\n' for s in source_lock if s['key'] in source_names)
        code_url = f"https://github.com/{spec['github_repository']}/tree/{args.code_revision}/{spec['github_directory']}"
        card = f'''---
{license_header}task_categories:
  - text-generation
pretty_name: "{title}"
size_categories:
  - 10K<n<100K
source_datasets:
{sources_yaml}tags:
  - math
  - reinforcement-learning
  - deduplicated
  - math-verify
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.parquet
---

# {title}

**{len(rows):,} questions. Only parser-compatible references are included.** Here, “verifiable” means every reference component parses with Math-Verify 0.8.0, with string fallback disabled. It does not mean that the answer has been independently proved correct or that grading model outputs is error-free.

{method}

## Preparation

See [PREPARATION.md](PREPARATION.md) for normalization, duplicate thresholds, conflict handling, benchmark screening, parser filtering, and limitations. [The complete preparation and publication code]({code_url}) is pinned to Git commit `{args.code_revision}`. `manifest.json` records counts and checksums; `source-revisions.json` pins the upstream snapshots.

Only this filtered training pool is published. Raw inputs, rejected examples, benchmark questions, unfiltered pools, and models are not included.

## Loading

```python
from datasets import load_dataset

data = load_dataset("{entry['repo_id']}", split="train")
```

The `train` split names the complete released pool; it is not a precomputed train/validation partition. Split by shared question groups across datasets before comparing experiments.

## Schema

- `id`: SHA-256 of the representative normalized prompt.
- `prompt`, `messages`: original representative question and its single user message.
- `answers`: ordered reference components; do not assume these are alternative accepted answers.
- `reference_answer_raw`: original source reference serialized as JSON.
- `datasets`, `source`, `source_records`: memberships and provenance; `source_records` is serialized as JSON.
- `verifiability`: parser-compatibility flags; independent label verification is false.

Separate reference solutions are not appended to prompts. Explicit answer/solution section markers trigger conservative exclusion. Native model chat templates, generated responses, rewards, and reasoning-length restrictions are not embedded in this dataset.

## Attribution and license

The upstream datasets are [Skywork/Skywork-OR1-RL-Data](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data) and [agentica-org/DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset), as applicable in `source-revisions.json`. DeepScaleR declares MIT; the pinned Skywork card does not declare a dataset license. This derivative does not assign a new license to upstream material. Preserve upstream attribution and consult the relevant source terms.
'''
        (folder / 'README.md').write_text(card)
        shutil.copyfile(code_root / 'PREPARATION.md', folder / 'PREPARATION.md')
        allowed = {'README.md', 'PREPARATION.md', 'manifest.json', 'source-revisions.json', 'data/train.parquet'}
        assert {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()} == allowed
        summary[name] = {'rows': len(rows), 'sha256': digest(parquet), 'bytes': parquet.stat().st_size}
    (args.output / 'export-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
