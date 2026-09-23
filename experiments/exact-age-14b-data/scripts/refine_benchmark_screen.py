import argparse
import json
import shutil
from pathlib import Path

from prepare import RULES, BenchmarkIndex, deduplicate, export, load_source, write_jsonl


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    report = json.loads((root / 'reports/preparation-report.json').read_text())
    archive = root / 'reports/initial-screen-audit'
    archive.mkdir(exist_ok=True)
    for path in [root / 'reports/preparation-report.json', root / 'reports/preparation-progress.json']:
        if path.exists():
            shutil.copy2(path, archive / path.name)
    index = BenchmarkIndex(read(root / 'data/benchmarks/all.jsonl'))
    cleaned = {}
    refinements = {}
    for name in ['skywork', 'deepscaler']:
        overlap_path = root / 'reports' / f'{name}-benchmark-overlap.jsonl'
        old_overlap = read(overlap_path)
        shutil.copy2(overlap_path, archive / overlap_path.name)
        rows = read(root / 'data/processed' / name / 'clean.jsonl')
        for row in rows:
            row['_members'] = [{'source_id': row['source_records'][0]['row_id'], 'prompt': row['prompt'], 'answers': row['answers']}]
        raw, _, _ = load_source(root, name, [])
        by_source_id = {r['source_records'][0]['row_id']: r for r in raw}
        retained_overlap, restored = [], []
        for record in old_overlap:
            members = [by_source_id[s['row_id']] for s in record['source_records']]
            matches = [{'source_id': r['source_records'][0]['row_id'], 'matches': index.match(r['prompt'])} for r in members if index.match(r['prompt'])]
            if matches:
                retained_overlap.append({**record, 'hits': matches})
            else:
                chosen = next(r for r in members if r['prompt'] == record['prompt'])
                item = dict(chosen)
                item['id'] = record['id']
                item['source_records'] = record['source_records']
                item['_members'] = [m for r in members for m in r['_members']]
                rows.append(item)
                restored.append({'id': item['id'], 'prompt': item['prompt'], 'previous_hits': record['hits'], 'reason': 'similar benchmark wording but different numeric parameters or uppercase entities'})
        rows.sort(key=lambda r: int(r['source_records'][0]['row_id'].rsplit(':', 1)[1]))
        cleaned[name] = rows
        report['versions'][name]['benchmark_overlap_unique_rows_removed'] = len(retained_overlap)
        report['versions'][name].update(export(root, name, rows))
        write_jsonl(overlap_path, retained_overlap)
        write_jsonl(root / 'reports' / f'{name}-restored-benchmark-variants.jsonl', restored)
        refinements[name] = {'old_possible_overlap_count': len(old_overlap), 'confirmed_under_revised_rules': len(retained_overlap), 'different_parameter_variants_restored': len(restored)}
    quarantine, receipts = [], []
    merged, profile = deduplicate(cleaned['skywork'] + cleaned['deepscaler'], 'merged', receipts, quarantine)
    report['versions']['merged'] = {'deduplication': profile, 'rows_with_both_dataset_memberships': sum(len(r['datasets']) == 2 for r in merged), **export(root, 'merged', merged)}
    write_jsonl(root / 'reports/merged-duplicate-receipts.jsonl', receipts)
    write_jsonl(root / 'data/quarantine/merged.jsonl', quarantine)
    report['rules'] = RULES
    report['benchmark_screen_refinement'] = refinements
    report['pipeline_note'] = 'Final screening requires matching numeric multisets and uppercase entities for near matches. Initial conservative screen receipts are archived for audit.'
    for rows in [*cleaned.values(), merged]:
        assert all(not index.match(r['prompt']) for r in rows)
    (root / 'reports/preparation-report.json').write_text(json.dumps(report, indent=2) + '\n')
    (root / 'reports/preparation-progress.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'refinement': refinements, 'counts': {k: v['clean_rows'] for k, v in report['versions'].items()}}), flush=True)


if __name__ == '__main__':
    main()
