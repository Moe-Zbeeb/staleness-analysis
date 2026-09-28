from dataclasses import dataclass
from functools import lru_cache
import hashlib
import re
from importlib.metadata import version
from pathlib import Path

from math_verify import LatexExtractionConfig, parse, verify
from math_verify.utils import timeout as bounded
from math_verify.errors import TimeoutException
from latex2sympy2_extended.latex2sympy2 import ConversionConfig, latex2sympy
from sympy import FiniteSet, Interval, S, Symbol, Tuple

EXTRACTION = LatexExtractionConfig(try_extract_without_anchor=False, boxed_match_priority=0)
POLICY = "strict-final-box-v4-bounded-prediction-verification"


class ReferenceRejected(ValueError):
    pass


def reward_identity():
    return {
        "policy": POLICY,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependencies": {
            name: version(name) for name in ("math-verify", "latex2sympy2-extended", "sympy", "antlr4-python3-runtime")
        },
    }


def last_complete_box(text):
    starts, offset = [], 0
    while (start := text.find("\\boxed{", offset)) >= 0:
        starts.append(start + 7)
        offset = start + 7
    for start in reversed(starts):
        depth = 1
        for index in range(start, len(text)):
            backslashes, previous = 0, index - 1
            while previous >= 0 and text[previous] == "\\":
                backslashes += 1
                previous -= 1
            if backslashes % 2:
                continue
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            if depth == 0:
                return text[start:index].strip()
    return ""


def normalize_reference(answer):
    text = answer.strip()
    pairs = (("\\[", "\\]"), ("\\(", "\\)"), ("$$", "$$"), ("$", "$"))
    while True:
        for left, right in pairs:
            if len(text) > len(left) + len(right) and text.startswith(left) and text.endswith(right):
                text = text[len(left) : -len(right)].strip()
                break
        else:
            break
    return last_complete_box(text) or text


@lru_cache(maxsize=50000)
def case_preserved(answer, timeout):
    @bounded(timeout_seconds=timeout)
    def convert():
        value = latex2sympy(
            normalize_reference(answer),
            normalization_config=EXTRACTION.normalization_config,
            conversion_config=ConversionConfig(lowercase_symbols=False),
        )
        return value.xreplace(
            {
                symbol: Symbol("".join(f"case{ord(char):06x}" for char in symbol.name), **symbol.assumptions0)
                for symbol in value.free_symbols
            }
        )

    return convert()


def structure_compatible(gold, predicted):
    if isinstance(gold, (Tuple, FiniteSet, Interval)) or isinstance(predicted, (Tuple, FiniteSet, Interval)):
        if isinstance(gold, FiniteSet) != isinstance(predicted, FiniteSet):
            return False
        if isinstance(gold, Tuple) and isinstance(predicted, Tuple):
            return len(gold) == len(predicted) and all(
                structure_compatible(left, right) for left, right in zip(gold, predicted, strict=True)
            )
        if isinstance(gold, FiniteSet) and isinstance(predicted, FiniteSet):
            return len(gold) == len(predicted)
        if isinstance(gold, (Tuple, Interval)) != isinstance(predicted, (Tuple, Interval)):
            return False
    return True


@dataclass(frozen=True)
class ClockAnswer:
    hour: int
    minute: int
    period: str | None = None


def clock_question(question):
    if re.search(r"\b(?:ratio|proportion)\b", question, re.IGNORECASE):
        return False
    if re.search(
        r"\b(?:clocks?|wristwatches?|watches|watch|noon|midnight)\b|o'clock|\btime of day\b|\bhh\s*:\s*mm\b"
        r"|\b(?:actual|correct|current|exact|earliest|arrival|local) time\b",
        question,
        re.IGNORECASE,
    ):
        return True
    if re.search(r"\b[ap]\.?m\.?(?=\W|$)", question, re.IGNORECASE):
        return True
    return bool(
        re.search(r"(?<![\d:])[0-2]?[0-9]\s*:\s*[0-5][0-9](?![\d:])", question)
        and re.search(r"\b(?:times?|when|arriv\w*|leav\w*|start\w*|finish\w*|meet\w*)\b", question, re.IGNORECASE)
    )


def parse_clock(answer):
    text = normalize_reference(answer)
    text = re.sub(r"\\(?:text|mathrm)\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\(?:[!,;: ]|qquad\b|quad\b|enspace\b|thinspace\b|medspace\b|thickspace\b)", "", text)
    match = re.fullmatch(
        r"([0-9]{1,2})\s*:\s*([0-9]{2})(?:\s*([AaPp])\.?[Mm]\.?|\s*\(\s*([AaPp])\.?[Mm]\.?\s*\))?",
        text.strip(),
    )
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    period = match.group(3) or match.group(4)
    if minute > 59 or hour > 23 or (period and not 1 <= hour <= 12):
        return None
    if period:
        hour = hour % 12 + (12 if period.lower() == "p" else 0)
    return ClockAnswer(hour, minute, "explicit" if period or hour > 12 or hour == 0 else None)


def clock_matches_reference(gold, predicted):
    if gold.period is None:
        return gold.hour % 12 == predicted.hour % 12 and gold.minute == predicted.minute
    return gold == predicted


def content_error(error):
    if isinstance(error, ValueError):
        return "thread" not in str(error).lower() and "signal" not in str(error).lower()
    if isinstance(error, (SyntaxError, ArithmeticError, NotImplementedError)):
        return True
    return type(error) is Exception and str(error).startswith(
        (
            "Nothing matched",
            "I expected",
            "I don't understand",
            "missing ",
            "Unrecognized ",
            "Row and col don's match",
            "Cannot perform modulo operation with a matrix",
            "Index out of bounds",
            "Expected expression for derivative",
            "Cannot apply postfix to derivative",
            "Cannot raise derivative to power",
        )
    )


def invalid_math_value(value):
    return bool(hasattr(value, "has") and value.has(S.NaN, S.ComplexInfinity))


def parse_math(text, timeout):
    return parse(
        f"\\boxed{{{text}}}",
        extraction_config=[EXTRACTION],
        fallback_mode="no_fallback",
        extraction_mode="first_match",
        parsing_timeout=timeout,
        raise_on_error=True,
    )


@lru_cache(maxsize=50000)
def parse_gold(answer, timeout, question=""):
    text = normalize_reference(answer)
    if ":" in text and re.search(r"\b(?:final|match) score\b", question, re.IGNORECASE):
        if not re.search(r"\b(?:ratio|proportion)\b", question, re.IGNORECASE):
            raise ReferenceRejected("Match-score answers require a separate grading contract")
    if ":" in text and re.search(r"\bmm\s*:\s*ss\b", question, re.IGNORECASE):
        raise ReferenceRejected("Minute-second duration answers require a separate grading contract")
    if ":" in text and clock_question(question):
        clock = parse_clock(text)
        if clock is None:
            raise ReferenceRejected("Clock reference must contain a complete, valid hour and two-digit minute")
        return [clock]
    try:
        parsed = parse_math(text, timeout)
        if not parsed or any(invalid_math_value(value) for value in parsed):
            raise ReferenceRejected("Reference answer is unparseable or contains undefined mathematical values")
        if not all(verify(value, value, timeout_seconds=timeout, raise_on_error=True) for value in parsed):
            raise ReferenceRejected("Reference answer cannot be verified against itself")
        if re.search("[A-Z]", text):
            preserved = case_preserved(text, timeout)
            if invalid_math_value(preserved) or not verify(
                preserved, preserved, timeout_seconds=timeout, raise_on_error=True
            ):
                raise ReferenceRejected("Reference cannot be verified while preserving symbol case")
    except ReferenceRejected:
        raise
    except Exception as error:
        if not content_error(error):
            raise
        raise ReferenceRejected("Reference cannot be verified by the pinned mathematical parser") from error
    return parsed


def grade_result(raw_completion, answer, truncated, truncated_reward, timeout, question="", reasoning_required=True):
    gold = parse_gold(answer, timeout, question)

    def result(reward, reason, extracted=""):
        return {"reward": float(reward), "reason": reason, "extracted": extracted, "policy": POLICY}

    if truncated and truncated_reward == "zero":
        return result(0, "truncated_zero_policy")
    reasoning_depth = 0
    for tag in re.finditer(r"</?think>", raw_completion):
        reasoning_depth = reasoning_depth + 1 if tag.group() == "<think>" else max(0, reasoning_depth - 1)
    if reasoning_depth:
        return result(0, "reopened_reasoning" if "</think>" in raw_completion else "missing_reasoning_close")
    if "</think>" in raw_completion:
        final = raw_completion.rsplit("</think>", 1)[1]
        if "<think>" in final:
            return result(0, "reopened_reasoning")
    elif reasoning_required or "<think>" in raw_completion:
        return result(0, "missing_reasoning_close")
    else:
        final = raw_completion
    boxed = last_complete_box(final)
    if not boxed:
        return result(0, "missing_complete_final_box")
    if isinstance(gold[0], ClockAnswer):
        predicted_clock = parse_clock(boxed)
        if predicted_clock is None:
            return result(0, "unsupported_clock_prediction", boxed)
        correct = clock_matches_reference(gold[0], predicted_clock)
        return result(correct, "correct" if correct else "not_verified_correct", boxed)
    attempts = []
    for budget in (timeout, 4 * timeout):
        try:
            outcome = verify_prediction(gold, boxed, answer, budget)
            outcome["verification_timeouts"] = attempts
            return outcome
        except TimeoutException as error:
            attempts.append({"timeout_seconds": budget, "message": str(error)[:256]})
    return {
        **result(0, "prediction_verification_timeout", boxed),
        "verification_status": "unverified",
        "verification_timeouts": attempts,
    }


def verify_prediction(gold, boxed, answer, timeout):
    def result(reward, reason, extracted=""):
        return {"reward": float(reward), "reason": reason, "extracted": extracted, "policy": POLICY}

    try:
        predicted = parse_math(boxed, timeout)
        if not predicted:
            return result(0, "unsupported_prediction", boxed)
        if any(invalid_math_value(value) for value in predicted):
            return result(0, "undefined_prediction", boxed)
        compatible = [(left, right) for left in gold for right in predicted if structure_compatible(left, right)]
        if not compatible:
            return result(0, "answer_structure_mismatch", boxed)
        correct = any(verify(left, right, timeout_seconds=timeout, raise_on_error=True) for left, right in compatible)
        if correct and re.search("[A-Z]", normalize_reference(answer) + boxed):
            try:
                predicted_case = case_preserved(boxed, timeout)
            except Exception as error:
                if not content_error(error):
                    raise
                return result(0, "unsupported_case_sensitive_prediction", boxed)
            if invalid_math_value(predicted_case):
                return result(0, "undefined_prediction", boxed)
            correct = verify(
                case_preserved(answer, timeout), predicted_case, timeout_seconds=timeout, raise_on_error=True
            )
            if not correct:
                return result(0, "symbol_case_mismatch", boxed)
    except Exception as error:
        if not content_error(error):
            raise
        return result(0, "unsupported_prediction", boxed)
    return result(correct, "correct" if correct else "not_verified_correct", boxed)


def grade(raw_completion, answer, truncated, truncated_reward, timeout, question="", reasoning_required=True):
    return grade_result(raw_completion, answer, truncated, truncated_reward, timeout, question, reasoning_required)[
        "reward"
    ]


@lru_cache(maxsize=1)
def tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def completion_tokens(trace):
    nodes = [node for node in trace.nodes if node.sampled]
    if len(nodes) != 1:
        raise ValueError("The study requires exactly one sampled assistant message")
    return [token for token, sampled in zip(nodes[0].token_ids, nodes[0].mask, strict=True) if sampled]
