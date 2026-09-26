"""Offline conservative v3 comparator; original extraction and grades stay frozen."""
import json
import re
import signal
from decimal import Decimal
from functools import lru_cache

from .extraction import final_answer

VERSION = 'math-sweep-final-v3.2'
EXACT_INTEGER = {'aime24', 'aime25', 'aime26', 'beyondaime'}


class Unsupported(ValueError):
    pass


class GradeTimeout(TimeoutError):
    pass


def alarm(*_):
    raise GradeTimeout('comparison time limit')


def normalize(value, unit=None):
    s = str(value).strip().replace(r'\$', '').replace('$', '').strip()
    for a, b in [(r'\left', ''), (r'\right', ''), (r'\dfrac', r'\frac'),
                 (r'\tfrac', r'\frac'), (r'\!', ''), (r'\,', ''),
                 (r'\;', ''), (r'\:', '')]:
        s = s.replace(a, b)
    s = re.sub(r'(?<!\\)\\ ', ' ', s)
    s = re.sub(r'\\(text|mathrm|mbox)\s*\{\s*~?\s*([^{}]*?)\s*\}', r'\\\1{\2}', s)
    s = re.sub(r'\\(?:quad|qquad)\b', ' ', s)
    s = re.sub(r'\\\\\s*\[[\d.]+(?:em|ex|pt|mm|cm)\]', lambda _: r'\\', s)
    s = re.sub(r'\\frac\s*([A-Za-z0-9])\s*([A-Za-z0-9])', r'\\frac{\1}{\2}', s)
    s = re.sub(r'\\frac\s*([A-Za-z0-9])\s*(\{[^{}]+\})', r'\\frac{\1}\2', s)
    s = re.sub(r'\\frac\s*(\{[^{}]+\})\s*([A-Za-z0-9])', r'\\frac\1{\2}', s)
    s = re.sub(r'\\sqrt\s*([A-Za-z0-9])', r'\\sqrt{\1}', s)
    # Brace spelling of a single-token subscript has no mathematical meaning.
    s = re.sub(r'_\s*(\\[A-Za-z]+|[A-Za-z0-9])(?![A-Za-z])', r'_{\1}', s)
    s = re.sub(r'(?<![A-Za-z])([0-9]+(?:\.[0-9]+)?)[eE]([+-]?[0-9]+)',
               lambda m: m[1] + r'\times10^{' + m[2] + '}', s)
    # Literal reference notation, never evaluate source code.
    s = re.sub(r'\bnp\.(arcsin|arccos|arctan|sin|cos|tan|exp|log|sqrt)\s*\(',
               lambda m: '\\' + ('ln' if m[1] == 'log' else m[1]) + '(', s)
    if unit:
        u = str(unit).strip('$')
        for suffix in sorted({u, r'\mathrm{' + u + '}', r'\text{' + u + '}'}, key=len, reverse=True):
            if s.endswith(suffix):
                s = s[:-len(suffix)].rstrip()
                break
    return s.strip()


def split_top(s):
    result, start, stack = [], 0, []
    pairs = {')': '(', ']': '[', '}': '{'}
    for i, c in enumerate(s):
        if c in '([{':
            stack.append(c)
        elif c in ')]}':
            if not stack:
                raise Unsupported('unbalanced delimiters')
            opening = stack.pop()
            if opening != pairs[c] and not (opening in '([' and c in ')]'):
                raise Unsupported('unbalanced delimiters')
        elif c in ',;' and not stack:
            result.append(s[start:i].strip().strip('$').strip())
            start = i + 1
    if stack:
        raise Unsupported('unbalanced delimiters')
    result.append(s[start:].strip().strip('$').strip())
    if not all(result):
        raise Unsupported('empty component')
    return result


def outer(s, left, right):
    if not s.startswith(left) or not s.endswith(right):
        return None
    inside = s[len(left):-len(right)]
    try:
        split_top(inside)
    except Unsupported:
        return None
    return inside


def structure(s, multiple=False):
    # These are separators between complete components, not permission to search
    # reasoning or to accept a matching subset of an answer.
    s = re.sub(r'\s*\\text\s*\{\s*and\s*\}\s*', ',', s)
    s = re.sub(r'\)\s+(?=\()', '),', s)
    m = re.fullmatch(r'\\begin\{([pbBvV]?matrix)\}(.*)\\end\{\1\}', s, re.S)
    if m:
        rows = [r.strip() for r in m[2].split(r'\\') if r.strip()]
        vals = [r.split('&') for r in rows]
        if not vals or len({len(r) for r in vals}) != 1:
            raise Unsupported('invalid matrix shape')
        # Determinant bars are not interchangeable with matrix parentheses.
        kind = 'determinant' if m[1] in {'vmatrix', 'Vmatrix'} else 'matrix'
        return kind, (len(vals), len(vals[0])), [x.strip() for r in vals for x in r]
    inside = outer(s, r'\{', r'\}')
    if inside is not None:
        vals = split_top(inside) if inside.strip() else []
        vals = [y for x in vals for y in ([x.replace(r'\pm', '+'), x.replace(r'\pm', '-')] if x.count(r'\pm') == 1 else [x])]
        return 'set', None, vals
    if len(s) > 2 and (s[0], s[-1]) in [('(', ']'), ('[', ')')]:
        inside = outer(s, s[0], s[-1])
        if inside is not None:
            vals = split_top(inside)
            if len(vals) == 2:
                return 'interval', (s[0], s[-1]), vals
    for left, right in [('(', ')'), ('[', ']')]:
        inside = outer(s, left, right)
        if inside is not None:
            vals = split_top(inside)
            if len(vals) > 1:
                return 'tuple' if left == '(' else 'bracketed', None, vals
    # Numeric commas are removed only in scalar context, never in sets/tuples.
    compact = re.sub(r'\s+', '', s)
    if not multiple and re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', compact):
        return 'scalar', None, [compact.replace(',', '')]
    vals = split_top(s)
    if len(vals) > 1:
        return 'multi' if multiple else 'sequence', None, vals
    if multiple and s.count(r'\pm') == 1:
        return 'multi', None, [s.replace(r'\pm', '+'), s.replace(r'\pm', '-')]
    return 'scalar', None, [s]


@lru_cache(maxsize=8192)
def parse_full(s, complex_i=False):
    import sympy as sp
    from antlr4 import Token
    from latex2sympy2_extended.latex2sympy2 import _Latex2Sympy, ConversionConfig
    from latex2sympy2_extended import NormalizationConfig
    from latex2sympy2_extended.math_normalization import normalize_latex

    # E with a subscript is an energy symbol, not Euler's constant. Preserve
    # case and subscript identity; do not rename arbitrary variables.
    s = re.sub(r'(?<![A-Za-z\\])([A-Za-z]|\\(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|pi|rho|sigma|tau|phi|psi|omega|Delta|Theta|Lambda|Sigma|Phi|Psi|Omega))_\{([^{}]+)\}',
               lambda m: r'\mathit{v' + m[1].encode().hex() + 'sub' + m[2].encode().hex() + '}', s)

    class WholeParser(_Latex2Sympy):
        def create_parser(self, value):
            self.used_parser = super().create_parser(value)
            return self.used_parser

    parser = WholeParser(None, None, False, config=ConversionConfig(
        lowercase_symbols=False, interpret_simple_eq_as_assignment=False,
        interpret_contains_as_eq=False, interpret_as_mixed_fractions=False))
    s = normalize_latex(s, NormalizationConfig(units=False, boxed='none', equations=False))
    try:
        value = parser.parse(s)
        if parser.used_parser.getCurrentToken().type != Token.EOF:
            raise Unsupported('unconsumed mathematical text')
    except (GradeTimeout, MemoryError, RecursionError):
        raise
    except Exception as exc:
        if isinstance(exc, Unsupported) or any(t in str(exc) for t in (
                'I expected', "I don't understand", 'Nothing matched', 'Cannot', 'Unrecognized', 'Could not', 'Invalid')):
            raise Unsupported(str(exc)[:200]) from exc
        raise
    if complex_i and hasattr(value, 'free_symbols'):
        value = value.xreplace({x: sp.I for x in value.free_symbols if str(x) == 'i'})
    return value


def poles(value):
    """Conservative domain guard: don't silently cancel variable denominators."""
    import sympy as sp
    result = set()
    for p in sp.preorder_traversal(value):
        if isinstance(p, sp.Pow) and p.exp.is_number and p.exp.is_negative and p.base.free_symbols:
            for factor in sp.Mul.make_args(sp.factor(p.base)):
                if factor.free_symbols:
                    base = factor.base if isinstance(factor, sp.Pow) else factor
                    # A-b and b-A exclude precisely the same zero set.
                    numerator = sp.fraction(sp.together(base))[0]
                    if numerator.could_extract_minus_sign():
                        numerator = -numerator
                    result.add(sp.srepr(numerator))
    return result


def scalar_equal(a, b, meta):
    import sympy as sp
    if re.sub(r'\s+', '', a) == re.sub(r'\s+', '', b):
        return True, 'normalized_identity'
    def word(s):
        m = re.fullmatch(r'\\(?:text|mbox|mathrm)\s*\{\s*([A-Za-z ]+)\s*\}', s)
        return m[1].strip() if m else s.strip()
    if re.fullmatch(r'[A-Za-z ]+', word(a)) and word(a) == word(b):
        return True, 'text_wrapper_identity'
    ga, pa = parse_full(a, bool(meta.get('complex_i'))), parse_full(b, bool(meta.get('complex_i')))
    for before, after in meta.get('symbol_aliases', {}).items():
        ga = ga.xreplace({v: sp.Symbol(after) for v in ga.free_symbols if str(v) == before})
        pa = pa.xreplace({v: sp.Symbol(after) for v in pa.free_symbols if str(v) == before})
    if meta.get('constant_case'):
        names = meta['constant_case']
        target = sp.Symbol(names[0])
        ga = ga.xreplace({v: target for v in ga.free_symbols if str(v) in names})
        pa = pa.xreplace({v: target for v in pa.free_symbols if str(v) in names})
    # Full parser can represent comma lists or logical statements. They cannot
    # enter a scalar comparison or math-verify's any-pair list comparison.
    if isinstance(ga, (sp.Set, sp.MatrixBase)) or isinstance(pa, (sp.Set, sp.MatrixBase)):
        return False, 'unsupported_scalar_structure'
    if isinstance(ga, sp.Equality) or isinstance(pa, sp.Equality):
        if not isinstance(ga, sp.Equality) or not isinstance(pa, sp.Equality):
            # A label already present in the reference, or a small frozen
            # question-specific label registry, can wrap the requested value.
            eq = ga if isinstance(ga, sp.Equality) else pa
            other = pa if isinstance(ga, sp.Equality) else ga
            simple_lhs = isinstance(eq.lhs, sp.Symbol) or isinstance(eq.lhs, sp.core.function.AppliedUndef)
            registered = any(
                eq.lhs == parse_full(normalize(label)) for label in meta.get('answer_labels', []))
            allowed = (isinstance(ga, sp.Equality) and simple_lhs) or registered
            if allowed and eq.lhs not in eq.rhs.free_symbols:
                delta = eq.rhs-other
                if poles(eq.rhs) == poles(other) and (sp.simplify(delta) == 0 or
                        (delta.has(sp.sinh, sp.cosh, sp.tanh) and sp.simplify(delta.rewrite(sp.exp)) == 0)):
                    return True, 'explicit_answer_label'
            return False, 'equation_expression_mismatch'
        g, p = ga.lhs - ga.rhs, pa.lhs - pa.rhs
        if poles(ga) != poles(pa):
            return False, 'equation_domain_guard'
        if sp.simplify(g-p) == 0 or sp.simplify(g+p) == 0:
            return True, 'equation_rearrangement'
        if p != 0:
            ratio = sp.simplify(g/p)
            if ratio.is_number and ratio.is_finite and ratio.is_zero is False:
                return True, 'equation_constant_multiple'
        return False, 'equation_inequality'
    if not isinstance(ga, sp.Expr) or not isinstance(pa, sp.Expr):
        return bool(ga == pa), 'literal_relation'
    if ga.is_number and pa.is_number:
        if sp.simplify(ga-pa) == 0:
            return True, 'exact_numeric_equality'
        if meta.get('benchmark') in EXACT_INTEGER or (
                meta.get('benchmark') != 'minerva' and not meta.get('error') and
                ga.is_integer is True and pa.is_integer is True):
            return False, 'exact_numeric_inequality'
        try:
            x, y = Decimal(str(ga.evalf(40))), Decimal(str(pa.evalf(40)))
            if not x.is_finite() or not y.is_finite():
                return False, 'nonfinite_numeric'
            tolerance = Decimal(str(meta['error'])) if meta.get('error') else abs(x)*Decimal(str(meta.get('relative_tolerance', '0.000001')))
            return abs(x-y) <= tolerance, 'existing_numeric_tolerance'
        except ArithmeticError:
            return False, 'nonreal_numeric_inequality'
    if poles(ga) != poles(pa):
        return False, 'expression_domain_guard'
    difference = ga-pa
    if sp.simplify(difference) == 0:
        return True, 'symbolic_identity'
    if difference.has(sp.sinh, sp.cosh, sp.tanh):
        return sp.simplify(difference.rewrite(sp.exp)) == 0, 'hyperbolic_identity'
    return False, 'symbolic_inequality'


def compare(gold, answer, meta):
    a, b = normalize(gold, meta.get('unit')), normalize(answer, meta.get('unit'))
    for unit in meta.get('optional_units', []):
        unit = normalize(unit)
        if unit and a.endswith(unit):
            a = a[:-len(unit)].strip()
        if unit and b.endswith(unit):
            b = b[:-len(unit)].strip()
    # The variable membership label is explicit in the reference; preserve
    # boundaries and both interval endpoints when comparing its bare interval.
    membership = re.fullmatch(r'([A-Za-z]|\\[A-Za-z]+)\s*\\in\s*(.+)', a)
    if membership and r'\in' not in b:
        a = membership[2]
    if not a or not b or max(len(a), len(b)) > 20000 or '__' in a+b or '\x00' in a+b:
        return False, 'unsupported_input'
    multi = meta.get('multiple', False) or meta.get('unordered_sequence', False)
    ak, ashape, av = structure(a, multi)
    bk, bshape, bv = structure(b, multi)
    component_units = meta.get('component_units')
    if component_units and len(av) == len(component_units):
        bound = {}
        for item in bv:
            for i, units in enumerate(component_units):
                for unit in units:
                    unit = normalize(unit)
                    if item.endswith(unit):
                        if i in bound:
                            return False, 'duplicate_unit_labeled_component'
                        bound[i] = item[:-len(unit)].strip()
                        break
        if bound:
            if len(bound) != len(av) or len(bv) != len(av):
                return False, 'incomplete_unit_labeled_components'
            child = {**meta, 'component_units': None, 'multiple': False, 'unordered_sequence': False}
            return all(compare(x, bound[i], child)[0] for i, x in enumerate(av)), 'complete_unit_labeled_components'
    labels = meta.get('component_labels')
    if labels and len(av) == len(labels) and ak in {'multi', 'sequence'}:
        labelled = [re.fullmatch(r'(.+?)\s*=\s*(.+)', x) for x in bv]
        if any(labelled):
            if not all(labelled) or len(bv) != len(labels):
                return False, 'incomplete_labeled_components'
            parsed_labels = [parse_full(normalize(label)) for label in labels]
            positions = {}
            for item in labelled:
                label = normalize(item[1])
                label = meta.get('label_aliases', {}).get(label, label)
                lhs = parse_full(label)
                if lhs not in parsed_labels or lhs in positions:
                    return False, 'unexpected_or_duplicate_label'
                positions[lhs] = item[2]
            child = {**meta, 'multiple': False, 'component_labels': None, 'unordered_sequence': False}
            return all(compare(x, positions[label], child)[0] for x, label in zip(av, parsed_labels)), 'complete_labeled_components'
        if bk == 'tuple':
            ak = bk = 'tuple'
    if ak == bk == 'scalar':
        return scalar_equal(av[0], bv[0], meta)
    # Explicit set braces may replace the benchmark's unordered multiple-answer
    # list. Parenthesized tuples retain order even when multiple=True.
    if {ak, bk} <= {'set', 'multi'}:
        ak = bk = 'set'
    elif {ak, bk} <= {'sequence', 'tuple'} or {ak, bk} <= {'multi', 'tuple'}:
        ak = bk = 'tuple'
    if ak != bk or ashape != bshape or len(av) != len(bv):
        return False, 'structure_or_cardinality_mismatch'
    child_meta = {**meta, 'multiple': False, 'unordered_sequence': False, 'component_labels': None, 'component_units': None}
    if ak in {'set', 'multi'}:
        # Find a complete bijection; greedy numeric matching can miss one.
        edges = [[j for j, y in enumerate(bv) if compare(x, y, child_meta)[0]] for x in av]
        used = {}
        def assign(i, seen):
            for j in edges[i]:
                if j in seen:
                    continue
                seen.add(j)
                if j not in used or assign(used[j], seen):
                    used[j] = i
                    return True
            return False
        return all(assign(i, set()) for i in range(len(av))), 'complete_unordered_components'
    return all(compare(x, y, child_meta)[0] for x, y in zip(av, bv)), 'complete_ordered_components'


def grade(payload):
    extracted = final_answer(payload['raw'])
    result = {k: extracted[k] for k in ['extracted', 'extraction_status']}
    result['grader_version'] = VERSION
    if extracted['extracted'] is None:
        return {**result, 'correct': False, 'status': 'no_final', 'reason': extracted['extraction_status']}
    gold = payload['gold']
    gold = ', '.join(gold) if isinstance(gold, list) else str(gold)
    meta = {**payload.get('meta', {}), 'benchmark': payload.get('benchmark', '')}
    # Only interpret i as sqrt(-1) when the frozen question explicitly invokes
    # complex numbers. No variable relabeling, guessed assumptions, or LLM judge.
    meta['complex_i'] = bool(re.search(r'\bcomplex\b|\bimaginary\b', payload.get('question', ''), re.I))
    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, 8)
    try:
        correct, reason = compare(gold, extracted['extracted'], meta)
        return {**result, 'correct': bool(correct), 'status': 'graded', 'reason': reason}
    except Unsupported as exc:
        return {**result, 'correct': False, 'status': 'unsupported', 'reason': str(exc)[:200]}
    except Exception as exc:
        return {**result, 'correct': None, 'status': 'grader_error', 'reason': type(exc).__name__ + ': ' + str(exc)[:200]}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    print(json.dumps(grade(json.load(__import__('sys').stdin)), ensure_ascii=False))
