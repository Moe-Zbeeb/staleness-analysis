import importlib.metadata
import json
import unittest

from common import ROOT, read_rows, validate_manifest, write_json
from grading import grade, reference


def main():
    validate_manifest()
    tests = unittest.defaultTestLoader.discover(str(ROOT), pattern='test_grading.py')
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    assert result.wasSuccessful()
    spec = json.loads((ROOT / 'spec.json').read_text())
    rows = read_rows(ROOT / 'inputs/samples.jsonl')
    assert len(rows) == 512 and len({r['key'] for r in rows}) == 512
    for dataset in spec['datasets']:
        assert sum(r['dataset'] == dataset for r in rows) == 256
    failures = []
    review = []
    for row in rows:
        assert len(row['answers']) == 1, row['key']
        if not reference(row['answers'][0]):
            review.append(row['key'])
            continue
        answer = row['answers'][0].strip().strip('$')
        text = answer if '\\boxed' in answer else '\\boxed{' + answer + '}'
        if not grade({'completion': text}, row['answers'])['correct']:
            failures.append(row['key'])
    assert not failures, failures
    assert review == spec['manual_review_reference_keys'], review
    all_slots = [r['key'] for slot in range(8) for r in rows[slot::8]]
    assert sorted(all_slots) == sorted(r['key'] for r in rows)
    assert len(rows) * len(spec['models']) * len(spec['budgets']) == 7680
    summary = {'status': 'passed', 'grader_tests': result.testsRun, 'reference_self_checks': len(rows) - len(review), 'manual_review_reference_keys': review, 'sample_count': len(rows), 'expected_responses': 7680, 'expected_cells': 30, 'math_verify_version': importlib.metadata.version('math-verify')}
    write_json(ROOT / 'validation.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
