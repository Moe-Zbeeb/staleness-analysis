from functools import lru_cache
import hashlib
from importlib.metadata import version
from pathlib import Path

from math_verify import LatexExtractionConfig, parse, verify

EXTRACTION = LatexExtractionConfig(try_extract_without_anchor=False, boxed_match_priority=0)
POLICY = "strict-final-box-v1"


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
def parse_gold(answer, timeout):
    text = normalize_reference(answer)
    parsed = parse(
        f"\\boxed{{{text}}}",
        extraction_config=[EXTRACTION],
        fallback_mode="no_fallback",
        extraction_mode="first_match",
        parsing_timeout=timeout,
        raise_on_error=True,
    )
    if not parsed:
        raise ReferenceRejected("Reference answer is not parseable with the pinned reward implementation")
    return parsed


def grade_result(raw_completion, answer, truncated, truncated_reward, timeout):
    gold = parse_gold(answer, timeout)

    def result(reward, reason, extracted=""):
        return {"reward": float(reward), "reason": reason, "extracted": extracted, "policy": POLICY}

    if truncated and truncated_reward == "zero":
        return result(0, "truncated_zero_policy")
    if "</think>" not in raw_completion:
        return result(0, "missing_reasoning_close")
    final = raw_completion.rsplit("</think>", 1)[1]
    if "<think>" in final:
        return result(0, "reopened_reasoning")
    boxed = last_complete_box(final)
    if not boxed:
        return result(0, "missing_complete_final_box")
    predicted = parse(
        f"\\boxed{{{boxed}}}",
        extraction_config=[EXTRACTION],
        fallback_mode="no_fallback",
        extraction_mode="first_match",
        parsing_timeout=timeout,
        raise_on_error=True,
    )
    if not predicted:
        return result(0, "unsupported_prediction", boxed)
    correct = verify(gold, predicted, timeout_seconds=timeout, raise_on_error=True)
    return result(correct, "correct" if correct else "not_verified_correct", boxed)


def grade(raw_completion, answer, truncated, truncated_reward, timeout):
    return grade_result(raw_completion, answer, truncated, truncated_reward, timeout)["reward"]


@lru_cache(maxsize=1)
def tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def completion_tokens(trace):
    nodes = [node for node in trace.nodes if node.sampled]
    if len(nodes) != 1:
        raise ValueError("The study requires exactly one sampled assistant message")
    return [token for token, sampled in zip(nodes[0].token_ids, nodes[0].mask, strict=True) if sampled]
