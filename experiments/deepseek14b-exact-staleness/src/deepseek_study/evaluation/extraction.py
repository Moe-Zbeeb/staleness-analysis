"""Frozen sweep final-answer extraction."""
import re

def boxes(text):
    values = []
    for match in re.finditer(r'\\(?:boxed|fbox)\s*\{', text):
        start = match.end()
        depth = 1
        end = start
        while end < len(text) and depth:
            if text[end] == '{' and (end == 0 or text[end-1] != '\\'):
                depth += 1
            elif text[end] == '}' and (end == 0 or text[end-1] != '\\'):
                depth -= 1
            end += 1
        if depth == 0:
            values.append(text[start:end-1].strip())
    return values

def final_answer(raw):
    # All four prompts begin a reasoning span. Never score a box still inside it.
    if '</think>' not in raw:
        return {'reasoning':raw, 'final':'', 'extracted':None, 'extraction_status':'no_final_channel'}
    reasoning, final = raw.rsplit('</think>', 1)
    answers = boxes(final)
    extracted = answers[-1] if answers and answers[-1].strip() else None
    return {'reasoning':reasoning, 'final':final, 'extracted':extracted,
            'extraction_status':'boxed' if extracted is not None else 'no_final_box'}

