import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from common import digest
from configure import configure
from freeze import freeze
from score import apply_manual, metrics


class WorkflowTests(unittest.TestCase):
    def test_completed_length_and_pending_denominator(self):
        rows = [{'correct': True, 'completion_tokens': 20, 'finish_reason': 'stop', 'answer_parseable': True, 'grade_status': 'correct'},
                {'correct': None, 'completion_tokens': 100, 'finish_reason': 'length', 'answer_parseable': False, 'grade_status': 'needs_reference_review'}]
        result = metrics(rows)
        self.assertEqual(result['accuracy_bounds'], [.5, 1])
        self.assertIsNone(result['accuracy'])
        self.assertEqual(result['mean_response_tokens'], 60)
        self.assertEqual(result['mean_finished_response_tokens'], 20)
        self.assertEqual(result['finished_count'], 1)
        self.assertIsNone(metrics(rows[1:])['mean_finished_response_tokens'])

    def test_manual_review_is_bound_to_completion(self):
        row = {'correct': None, 'completion': 'specific final answer'}
        review = {'correct': True, 'rationale': 'Reference-equivalent final answer', 'completion_sha256': hashlib.sha256(row['completion'].encode()).hexdigest()}
        other = dict(row, completion='another answer')
        with self.assertRaises(AssertionError):
            apply_manual(other, review)
        apply_manual(row, review)
        self.assertTrue(row['correct'])
        self.assertEqual(row['grade_status'], 'manual_correct')
        with self.assertRaises(AssertionError):
            apply_manual(row, review)

    def test_configure_paths_and_refuse_existing_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            spec = {'datasets': ['skywork'], 'models': [{'tag': 'deepseek'}]}
            (root / 'experiment.json').write_text(json.dumps(spec))
            source = root / 'dataset/data/processed/skywork/train.parquet'
            source.parent.mkdir(parents=True)
            source.touch()
            target = root / 'models/deepseek'
            target.mkdir(parents=True)
            for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
                (target / name).write_text('{}')
            result = configure(root, root / 'dataset', root / 'models', sys.executable)
            self.assertEqual(result['models'][0]['target'], str(target))
            self.assertEqual(result['models'][0]['tokenizer_path'], str(root / 'inputs/tokenizers/deepseek'))
            with self.assertRaises(FileExistsError):
                configure(root, root / 'dataset', root / 'models', sys.executable)

    def test_freeze_detects_changes_and_does_not_refreeze(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            spec = {'models': [{'tag': 'deepseek'}]}
            (root / 'spec.json').write_text(json.dumps(spec))
            check = {'status': 'passed', 'models': {'deepseek': {'status': 'passed'}}}
            for name in ['tokenizer-validation.json', 'model-validation.json']:
                (root / name).write_text(json.dumps(check))
            for name in ['experiment.json', 'datasets.lock.json', 'preparation.json', 'runtime-versions.json']:
                (root / name).write_text('{}')
            (root / 'inputs').mkdir()
            (root / 'inputs/samples.jsonl').write_text('{}\n')
            (root / 'inputs/deepseek.jsonl').write_text('{}\n')
            (root / 'inputs/._samples.jsonl').write_text('macOS metadata')
            manifest = freeze(root)
            self.assertNotIn('inputs/._samples.jsonl', manifest['files'])
            (root / 'inputs/deepseek.jsonl').write_text('{"changed": true}\n')
            self.assertNotEqual(digest(root / 'inputs/deepseek.jsonl'), manifest['files']['inputs/deepseek.jsonl'])
            with self.assertRaises(FileExistsError):
                freeze(root)


if __name__ == '__main__':
    unittest.main()
