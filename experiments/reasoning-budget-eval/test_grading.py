import unittest

from grading import grade


class GradingTests(unittest.TestCase):
    def check(self, text, answer, expected, prefilled=False):
        result = grade({'completion': text, 'prefilled_think': prefilled}, [answer])
        self.assertEqual(result['correct'], expected, result)

    def test_equivalent_fraction(self):
        self.check(r'<think>work</think> The final answer is $\boxed{0.5}$.', r'\frac{1}{2}', True)

    def test_wrong_final_does_not_match_reasoning(self):
        self.check(r'<think>Maybe \boxed{2}.</think> Final answer: \boxed{3}', '2', False)

    def test_unfinished_prefilled_reasoning(self):
        self.check(r'We found \boxed{2} but need to check', '2', False, True)

    def test_unfinished_generated_reasoning(self):
        self.check(r'<think>We found \boxed{2}', '2', False)

    def test_direct_answer(self):
        self.check(r'The final answer is \boxed{42}.', '42', True)

    def test_nested_box(self):
        self.check(r'<think>x</think>\boxed{\frac{\sqrt{4}}{4}}', r'\frac{1}{2}', True)

    def test_ordered_tuple(self):
        self.check(r'\boxed{(2, 3)}', '(2,3)', True)
        self.check(r'\boxed{(3, 2)}', '(2,3)', False)

    def test_missing_answer(self):
        self.check('<think>work</think>', '2', False)

    def test_distinct_multiple_references_are_not_or(self):
        with self.assertRaises(AssertionError):
            grade({'completion': r'\boxed{2}'}, ['2', '3'])

    def test_complex_reference_cannot_match_a_fraction_fragment(self):
        answer = r'\max \left(a_{1}, \ldots, a_{n}, \frac{1}{2} \sum_{i=1}^{n}\left|a_{i}-a_{i+1}\right|\right)'
        result = grade({'completion': r'\boxed{\frac{1}{2}}'}, [answer])
        self.assertIsNone(result['correct'])
        self.assertEqual(result['grade_status'], 'needs_reference_review')


if __name__ == '__main__':
    unittest.main()
