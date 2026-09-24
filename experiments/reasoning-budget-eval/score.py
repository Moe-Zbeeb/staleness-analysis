import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import math
import statistics

from common import ROOT, digest, read_rows, validate_manifest, write_json
from grading import grade, reference


def evaluate(item):
    record, row = item
    return {**record, 'reference_answers': row['answers'], **grade(record, row['answers'])}


def wilson(correct, count):
    z = 1.959963984540054
    p = correct / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return [center - half, center + half]


def metrics(rows):
    count = len(rows)
    correct = sum(r['correct'] is True for r in rows)
    unresolved = sum(r['correct'] is None for r in rows)
    lengths = sorted(r['completion_tokens'] for r in rows)
    finished = [r['completion_tokens'] for r in rows if r['finish_reason'] == 'stop']
    return {'finished_count': len(finished), 'mean_finished_response_tokens': statistics.mean(finished) if finished else None, 'n': count, 'correct': correct, 'pending_review': unresolved, 'accuracy': correct / count if not unresolved else None, 'accuracy_bounds': [correct / count, (correct + unresolved) / count], 'accuracy_ci95_wilson': wilson(correct, count) if not unresolved else None, 'mean_response_tokens': statistics.mean(lengths), 'median_response_tokens': statistics.median(lengths), 'p90_response_tokens': lengths[math.ceil(.9 * count) - 1], 'total_response_tokens': sum(lengths), 'budget_hit_count': sum(r['finish_reason'] == 'length' for r in rows), 'budget_hit_rate': sum(r['finish_reason'] == 'length' for r in rows) / count, 'answer_parse_rate': sum(r['answer_parseable'] for r in rows) / count, 'grade_status_counts': dict(Counter(r['grade_status'] for r in rows))}


def apply_manual(row, review):
    assert row['correct'] is None, 'Only unresolved rows can receive a manual grade'
    assert isinstance(review['correct'], bool) and review['rationale']
    assert review['completion_sha256'] == hashlib.sha256(row['completion'].encode()).hexdigest(), 'Manual grade belongs to a different response'
    row.update(correct=review['correct'], grade_status='manual_correct' if review['correct'] else 'manual_incorrect', manual_review=review)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    validate_manifest()
    manifest_hash = digest(ROOT / 'manifest.json')
    spec = json.loads((ROOT / 'spec.json').read_text())
    samples = {r['key']: r for r in read_rows(ROOT / 'inputs/samples.jsonl')}
    for row in samples.values():
        assert len(row['answers']) == 1 and (reference(row['answers'][0]) or row['key'] in spec['manual_review_reference_keys']), row['key']
    records = []
    output = ROOT / 'outputs' / args.model
    for budget in spec['budgets']:
        seen = set()
        for path in sorted((output / str(budget)).glob('worker-*/batch-*.json')):
            saved = json.loads(path.read_text())
            assert saved['manifest_sha256'] == manifest_hash
            for row in saved['records']:
                assert row['key'] in samples and row['key'] not in seen
                assert row['model'] == args.model and row['budget'] == budget
                assert row['completion_tokens'] == len(row['completion_ids']) <= budget
                assert row['seed'] == samples[row['key']]['seed']
                seen.add(row['key'])
                records.append(row)
        assert seen == set(samples), (args.model, budget, len(seen))
    with ProcessPoolExecutor(max_workers=8) as pool:
        scored = list(pool.map(evaluate, [(r, samples[r['key']]) for r in records], chunksize=8))
    manual_path = output / 'manual-grades.json'
    manual = json.loads(manual_path.read_text()) if manual_path.exists() else {}
    for row in scored:
        key = row['key'] + ':' + str(row['budget'])
        if row['correct'] is None and key in manual:
            review = manual[key]
            apply_manual(row, review)
    path = output / 'scored.jsonl'
    temp = path.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in scored))
    temp.replace(path)
    results = []
    for dataset in spec['datasets']:
        for budget in spec['budgets']:
            group = [r for r in scored if r['dataset'] == dataset and r['budget'] == budget]
            assert len(group) == spec['sample_size']
            results.append({'model': args.model, 'dataset': dataset, 'budget': budget, **metrics(group)})
    summary = {'status': 'complete', 'model': args.model, 'responses': len(scored), 'pending_review': sum(r['correct'] is None for r in scored), 'manual_grades_sha256': digest(manual_path) if manual else None, 'manifest_sha256': manifest_hash, 'scored_sha256': digest(path), 'math_verify_version': importlib.metadata.version('math-verify'), 'results': results}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
