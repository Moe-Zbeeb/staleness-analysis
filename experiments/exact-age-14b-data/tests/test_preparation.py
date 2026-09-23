import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare import BenchmarkIndex, answer_values, deduplicate, near_duplicate, normalize, equivalent_answers, parsed_answer, prompt_leakage_marker


class PreparationTests(unittest.TestCase):
    def row(self, dataset, question, answer):
        return {'id': dataset + question, 'prompt': question, 'answers': [answer], 'datasets': [dataset], 'source_records': [{'dataset': dataset, 'row_id': dataset + question}], '_members': [{'source_id': dataset + question, 'prompt': question, 'answers': [answer]}]}

    def test_math_spacing_is_ignored(self):
        self.assertEqual(normalize('What is $2+2$?'), normalize('What is 2 + 2.'))

    def test_numeric_changes_are_not_duplicates(self):
        a = 'A long mathematical question asks for the total number of ways to choose exactly 4 objects out of a set of 100 distinct objects.'
        b = a.replace('4 objects', '5 objects')
        self.assertFalse(near_duplicate(normalize(a), normalize(b)))

    def test_operator_changes_are_not_duplicates(self):
        self.assertNotEqual(normalize('Find x if x + 2 = 7.'), normalize('Find x if x - 2 = 7.'))

    def test_layout_duplicates_collapse(self):
        rows = [self.row('skywork', 'Find $x$ if $x+2=7$.', '5'), self.row('skywork', '  Find x if x+2=7. ', '5')]
        clean, summary = deduplicate(rows, 'test', [], [])
        self.assertEqual(len(clean), 1)
        self.assertEqual(summary['redundant_rows'], 1)

    def test_conflicting_answers_quarantine_entire_group(self):
        quarantine = []
        rows = [self.row('skywork', 'Find x if x+2=7.', '5'), self.row('skywork', 'Find x if x+2=7.', '6')]
        clean, summary = deduplicate(rows, 'test', [], quarantine)
        self.assertEqual(clean, [])
        self.assertEqual(summary['answer_conflict_rows_quarantined'], 2)
        self.assertEqual(len(quarantine), 1)

    def test_merge_does_not_modify_standalone_membership(self):
        left, _ = deduplicate([self.row('skywork', 'Find x if x+2=7.', '5')], 'skywork', [], [])
        right, _ = deduplicate([self.row('deepscaler', 'Find x if x+2=7.', '5')], 'deepscaler', [], [])
        merged, _ = deduplicate(left + right, 'merged', [], [])
        self.assertEqual(len(left), 1)
        self.assertEqual(len(right), 1)
        self.assertEqual(len(merged), 1)
        self.assertEqual(left[0]['datasets'], ['skywork'])
        self.assertEqual(merged[0]['datasets'], ['deepscaler', 'skywork'])

    def test_benchmark_overlap_with_layout_changes(self):
        index = BenchmarkIndex([{'id': 'test:0', 'prompt': 'Find $x$ if $x+2=7$.'}])
        self.assertTrue(index.match('Find x if x+2=7.'))
        self.assertFalse(index.match('Find x if x+3=9.'))

    def test_mathematical_equivalence_is_supported(self):
        self.assertTrue(parsed_answer(r'\frac{1}{2}'))
        self.assertTrue(equivalent_answers(['0.5'], [r'\frac{1}{2}']))

    def test_reviewed_distinct_variant(self):
        index = BenchmarkIndex([{'id': 'math500:115', 'prompt': 'Determine the number of ways to arrange the letters of the word ELLIPSE.'}])
        self.assertFalse(index.match('Determine the number of ways to arrange the letters of the word PROOF.'))
        self.assertTrue(index.match('Determine the number of ways to arrange the letters of the word ELLIPSE.'))

    def test_explicit_solution_markers(self):
        self.assertTrue(prompt_leakage_marker('What is 2+2? Answer: 4'))
        self.assertTrue(prompt_leakage_marker('Find n. [hide=Solution] n=4'))
        self.assertFalse(prompt_leakage_marker('Give your answer. Express it as an integer.'))
        self.assertFalse(prompt_leakage_marker('The equation has two solutions. Find their sum.'))

    def test_reference_list_preserves_components(self):
        self.assertEqual(answer_values('["1", "2"]', decode_json=True), ['1', '2'])
        self.assertEqual(answer_values('[1, 2]', decode_json=False), ['[1, 2]'])


if __name__ == '__main__':
    unittest.main()
