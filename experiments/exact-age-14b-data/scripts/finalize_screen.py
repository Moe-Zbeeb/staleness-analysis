import argparse
import json
import shutil
from pathlib import Path

from prepare import RULES, BenchmarkIndex, deduplicate, export, load_source, prompt_leakage_marker, write_jsonl


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    archive = root / 'reports/parameter-screen-audit'
    archive.mkdir(exist_ok=False)
    for path in (root / 'reports').glob('*'):
        if path.is_file():
            shutil.copy2(path, archive / path.name)
    report = json.loads((root / 'reports/preparation-report.json').read_text())
    index = BenchmarkIndex(read(root / 'data/benchmarks/all.jsonl'))
    cleaned = {}
    for name in ['skywork', 'deepscaler']:
        raw_quarantine = []
        raw, raw_profile, _ = load_source(root, name, raw_quarantine)
        by_source = {r['source_records'][0]['row_id']: r for r in raw}
        leaked = {r['source_id']: r for r in raw_quarantine if r['reason'] == 'suspected_answer_or_solution_section_in_prompt'}
        existing_overlap = read(root / 'reports' / f'{name}-benchmark-overlap.jsonl')
        kept, overlap, leakage = [], [], []
        for row in read(root / 'data/processed' / name / 'clean.jsonl'):
            source_ids = [s['row_id'] for s in row['source_records']]
            if any(s in leaked for s in source_ids):
                leakage.append({'reason': 'suspected_answer_or_solution_section_in_prompt', 'row': row, 'flagged_source_ids': [s for s in source_ids if s in leaked]})
                continue
            members = [by_source[s] for s in source_ids]
            hits = [{'source_id': r['source_records'][0]['row_id'], 'matches': index.match(r['prompt'])} for r in members if index.match(r['prompt'])]
            if hits:
                overlap.append({'id': row['id'], 'prompt': row['prompt'], 'source_records': row['source_records'], 'hits': hits})
                continue
            row['_members'] = [m for r in members for m in r['_members']]
            kept.append(row)
        cleaned[name] = kept
        combined_overlap = existing_overlap + overlap
        write_jsonl(root / 'reports' / f'{name}-benchmark-overlap.jsonl', combined_overlap)
        write_jsonl(root / 'data/quarantine' / f'{name}-prompt-leakage.jsonl', leakage)
        report['versions'][name]['benchmark_overlap_unique_rows_removed'] = len(combined_overlap)
        report['versions'][name]['suspected_prompt_leakage_unique_rows_removed'] = len(leakage)
        report['versions'][name]['raw_suspected_prompt_leakage_rows'] = len(leaked)
        report['versions'][name].update(export(root, name, kept))
        print(json.dumps({'stage': 'final_screen', 'dataset': name, 'kept': len(kept), 'possible_overlap_removed': len(combined_overlap), 'suspected_prompt_leakage_removed': len(leakage)}), flush=True)
    quarantine, receipts = [], []
    merged, profile = deduplicate(cleaned['skywork'] + cleaned['deepscaler'], 'merged', receipts, quarantine)
    report['versions']['merged'] = {'deduplication': profile, 'rows_with_both_dataset_memberships': sum(len(r['datasets']) == 2 for r in merged), **export(root, 'merged', merged)}
    write_jsonl(root / 'reports/merged-duplicate-receipts.jsonl', receipts)
    write_jsonl(root / 'data/quarantine/merged.jsonl', quarantine)
    report['rules'] = {**RULES, 'prompt_leakage': 'Quarantine groups with explicit Answer/Solution section markers; conservative suspected leakage, not a semantic proof.'}
    report['pipeline_note'] = 'Final output uses conservative lexical benchmark screening with inspected distinct-variant pair exceptions. Prior screening iterations are archived. No numeric/uppercase-equality gate remains.'
    report['benchmark_screen_refinement']['final_reviewed_pairs'] = len(json.loads((root / 'configs/benchmark-reviewed-pairs.json').read_text()))
    report['validation']['no_detected_explicit_answer_solution_sections'] = True
    for rows in [*cleaned.values(), merged]:
        assert all(not index.match(r['prompt']) and not prompt_leakage_marker(r['prompt']) for r in rows)
    (root / 'reports/preparation-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'counts': {k: v['clean_rows'] for k,v in report['versions'].items()}}), flush=True)


if __name__ == '__main__':
    main()
