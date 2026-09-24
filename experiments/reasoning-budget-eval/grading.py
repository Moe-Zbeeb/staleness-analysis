from functools import lru_cache

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify


@lru_cache(maxsize=1024)
def reference(value):
    text = value.strip().strip('$')
    if '\\boxed' not in text:
        text = '\\boxed{' + text + '}'
    return parse(text, extraction_config=[LatexExtractionConfig(boxed_match_priority=0)], fallback_mode='no_fallback', extraction_mode='first_match', parsing_timeout=5)


def final_section(text, prefilled_think=False):
    if '</think>' in text:
        return text.rsplit('</think>', 1)[1].strip(), True
    if prefilled_think or '<think>' in text:
        return '', False
    return text.strip(), True


def grade(record, answers):
    assert len(answers) == 1, 'This sampled evaluation requires exactly one stored reference field'
    gold = reference(answers[0])
    final, finished = final_section(record['completion'], record.get('prefilled_think', False))
    if not gold:
        return {'correct': None, 'grade_status': 'needs_reference_review', 'answer_parseable': False, 'final_section_present': bool(final), 'parsed_prediction': [], 'parsed_reference': []}
    if not finished:
        return {'correct': False, 'grade_status': 'unfinished_reasoning', 'answer_parseable': False, 'final_section_present': False, 'parsed_prediction': [], 'parsed_reference': [str(g) for g in gold]}
    target = parse(final, extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()], fallback_mode='no_fallback', extraction_mode='first_match', parsing_timeout=5) if final else []
    correct = bool(target) and bool(verify(gold, target, timeout_seconds=5))
    return {'correct': correct, 'grade_status': 'correct' if correct else 'incorrect' if target else 'unparseable_answer', 'answer_parseable': bool(target), 'final_section_present': bool(final), 'parsed_prediction': [str(p) for p in target], 'parsed_reference': [str(g) for g in gold]}
