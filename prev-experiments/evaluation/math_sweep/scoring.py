import hashlib
import importlib.metadata
import importlib.util
import math
import random
import re
import statistics
from collections import defaultdict
from functools import lru_cache
from pathlib import Path


GRADER_PATH = Path(__file__).resolve().parents[2] / "packages/prime-rl-staleness/src/exact_math.py"
GRADER_SHA256 = "43d86f8ba5b399d7b489a3bb52341b95d472a8f98278486cbd2ef8cea03285f2"
FINAL_ANSWER = re.compile(
    r"(?i)^(?:(?:thus|therefore|hence)[,:]?\s+)?(?:the\s+)?"
    r"(?:final\s+)?answer\s*(?:is\s*[:=]?|[:=])\s*"
    r"(?P<answer>.+?)\s*(?:[.!])?\s*$"
)
THINK_TAG = re.compile(r"<(/?)think>", re.IGNORECASE)
MATH_CLOSERS = ("$$", "$", r"\)", r"\]", "`")
TERMINAL_WRAPPER = re.compile(r"(?:\$\$|\$|\\\)|\\\]|`)?[.!]?\s*$")


def grader_provenance():
    actual = hashlib.sha256(GRADER_PATH.read_bytes()).hexdigest()
    if actual != GRADER_SHA256:
        raise ValueError("Frozen exact_math.py hash mismatch")
    return {"path": str(GRADER_PATH), "sha256": actual, "timeout_seconds": 5}


@lru_cache(maxsize=1)
def _load_grader():
    grader_provenance()
    spec = importlib.util.spec_from_file_location("math_sweep_frozen_exact_math", GRADER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the frozen exact-math grader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _visible_text(text, thinking_open=False):
    opened = bool(thinking_open)
    last_close = 0
    for match in THINK_TAG.finditer(text):
        opened = not bool(match.group(1))
        if not opened:
            last_close = match.end()
    return (text if opened else text[last_close:]).strip(), opened


def decode_completion(token_ids, tokenizer, eos_ids, thinking_open=False):
    ids = list(token_ids)
    stops = set(eos_ids)
    if not stops or any(type(token) is not int or token < 0 for token in [*ids, *stops]):
        raise ValueError("Token and EOS IDs must be nonnegative integers with nonempty EOS IDs")
    eos_positions = [index for index, token in enumerate(ids) if token in stops]
    if eos_positions and eos_positions != [len(ids) - 1]:
        raise ValueError("Tokens appeared after EOS")
    content = ids[:-1] if eos_positions else ids
    kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    raw_text = tokenizer.decode(ids, **kwargs)
    visible, unfinished = _visible_text(tokenizer.decode(content, **kwargs), thinking_open)
    return {
        "raw_text": raw_text,
        "text": visible,
        "eos_seen": bool(eos_positions),
        "unfinished_thinking": unfinished,
    }


def _terminal_syntax(text):
    visible, unfinished = _visible_text(text)
    if unfinished:
        return False
    lines = [line.strip() for line in visible.splitlines() if line.strip()]
    if lines:
        closing = lines[-1]
        if closing.endswith((".", "!")):
            closing = closing[:-1].rstrip()
        if closing in MATH_CLOSERS:
            lines.pop()
    terminal = lines[-1] if lines else ""
    if FINAL_ANSWER.fullmatch(terminal):
        return True
    candidate = TERMINAL_WRAPPER.sub("", terminal).rstrip()
    start = candidate.rfind(r"\boxed{")
    if start < 0:
        return False
    depth = 1
    for index in range(start + len(r"\boxed{"), len(candidate)):
        depth += (candidate[index] == "{") - (candidate[index] == "}")
        if depth == 0:
            return index == len(candidate) - 1
    return False


def grade(text, answers, unfinished_thinking=False):
    if not isinstance(text, str) or not isinstance(answers, list) or not answers:
        raise ValueError("Grading requires text and a nonempty list of gold answers")
    if any(not isinstance(answer, str) or not answer.strip() for answer in answers):
        raise ValueError("Gold answers must be nonempty strings")
    _, detected_unfinished = _visible_text(text)
    if unfinished_thinking or detected_unfinished:
        return {"correct": False, "terminal_syntax": False, "status": "unfinished_thinking"}
    terminal = _terminal_syntax(text)
    if not terminal:
        return {"correct": False, "terminal_syntax": False, "status": "no_terminal_answer"}
    module = _load_grader()
    correct = any(module.verify_terminal_answer(text, answer, 5) == 1.0 for answer in answers)
    return {"correct": correct, "terminal_syntax": True, "status": "scored"}


def validate_grader():
    provenance = grader_provenance()
    two_roots = r"\frac{-1+\sqrt{17}}{2}, \frac{-1-\sqrt{17}}{2}"
    four_roots = r"2,3,\frac{3+i\sqrt{3}}{2},\frac{3-i\sqrt{3}}{2}"
    fixtures = [
        ("fraction", r"\frac{1}{576}", "1/576", True),
        ("unordered_two_roots", two_roots, r"\frac{-1-\sqrt{17}}{2},\frac{-1+\sqrt{17}}{2}", True),
        ("missing_second_root", two_roots, r"\frac{-1+\sqrt{17}}{2}", False),
        ("unordered_four_complex_roots", four_roots, r"\frac{3-i\sqrt{3}}{2},2,\frac{3+i\sqrt{3}}{2},3", True),
        ("missing_complex_roots", four_roots, "2", False),
        ("factorial", r"2^{25} \cdot 26!", r"2^{25} \cdot 26!", True),
        ("degree_annotation", r"74^\circ", "74", True),
        ("large_integer_expression", "2024^{4046}", "2024^{4046}", True),
        ("ordered_tuple", "(1,2)", "(2,1)", False),
        ("interval_endpoint", "[0,1)", "[0,1]", False),
    ]
    cases = []
    for name, gold, candidate, expected in fixtures:
        result = grade(rf"\boxed{{{candidate}}}", [gold])
        if result["correct"] != expected or result["status"] != "scored":
            raise ValueError(f"Frozen grader regression failed: {name}: {result}")
        cases.append({"name": name, "gold": gold, "prediction": candidate, "expected_correct": expected, "correct": result["correct"]})
    return {
        "success": True,
        "status": "passed",
        "grader": provenance,
        "versions": {name: importlib.metadata.version(name) for name in ("verifiers", "math-verify", "sympy", "latex2sympy2-extended")},
        "cases": cases,
    }


def pass_at_k(n, c, k):
    if any(type(value) is not int for value in (n, c, k)):
        raise ValueError("n, c and k must be integers")
    if n < 1 or not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("pass@k requires n >= 1, 0 <= c <= n and 1 <= k <= n")
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _interval(values, bootstrap_samples, rng):
    point = statistics.fmean(values)
    if bootstrap_samples == 0:
        return None
    if min(values) == max(values):
        return [point, point]
    draws = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(bootstrap_samples))
    def quantile(q):
        index = q * (len(draws) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        return draws[lower] + (draws[upper] - draws[lower]) * (index - lower)
    return [quantile(0.025), quantile(0.975)]


def _validate_record(record):
    for key in ("model_id", "family", "profile", "benchmark", "source_id"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"Record requires nonempty string {key}")
    if "staleness" not in record or (
        record["staleness"] is not None
        and (type(record["staleness"]) is not int or record["staleness"] < 0)
    ):
        raise ValueError("Record staleness must be a nonnegative integer or null for the starting model")
    for key in ("budget", "sample_index", "token_count"):
        if type(record.get(key)) is not int or record[key] < (1 if key == "budget" else 0):
            raise ValueError(f"Invalid {key}")
    if record["token_count"] > record["budget"]:
        raise ValueError("Completion token count exceeds output budget")
    for key in ("correct", "truncated", "terminal_syntax"):
        if type(record.get(key)) is not bool:
            raise ValueError(f"Record requires boolean {key}")
    if record["correct"] and not record["terminal_syntax"]:
        raise ValueError("A correct answer must have terminal syntax")
    if "heldout" in record and type(record["heldout"]) is not bool:
        raise ValueError("heldout must be boolean when present")


def summarize(records, bootstrap_samples=2000, seed=42):
    if type(bootstrap_samples) is not int or bootstrap_samples < 0 or type(seed) is not int:
        raise ValueError("Bootstrap samples must be a nonnegative integer and seed an integer")
    records = list(records)
    if not records:
        raise ValueError("No validated evaluation records supplied")
    groups = defaultdict(list)
    identities = {}
    seen = set()
    for record in records:
        _validate_record(record)
        identity = (record["family"], record["staleness"])
        previous = identities.setdefault(record["model_id"], identity)
        if identity != previous:
            raise ValueError("Model family or staleness changed within records")
        key = tuple(record[name] for name in ("model_id", "profile", "benchmark", "budget"))
        response_key = (*key, record["source_id"], record["sample_index"])
        if response_key in seen:
            raise ValueError(f"Duplicate evaluation response: {response_key}")
        seen.add(response_key)
        groups[key].append(record)
    rng = random.Random(seed)
    cells = []
    problem_results = {}
    grouped_records = {}
    for key in sorted(groups):
        selected = sorted(groups[key], key=lambda item: (item["source_id"], item["sample_index"]))
        by_problem = defaultdict(list)
        for record in selected:
            by_problem[record["source_id"]].append(record)
        sample_counts = {len(items) for items in by_problem.values()}
        if len(sample_counts) != 1:
            raise ValueError(f"Unequal samples per problem in {key}")
        n = sample_counts.pop()
        if any([row["sample_index"] for row in items] != list(range(n)) for items in by_problem.values()):
            raise ValueError(f"Missing or noncontiguous sample indices in {key}")
        for items in by_problem.values():
            for name in ("prompt_sha256", "answers", "heldout", "comparison_sha256"):
                if any(row.get(name) != items[0].get(name) for row in items):
                    raise ValueError(f"Problem metadata changed between samples: {name}")
        outcomes = {source: sum(row["correct"] for row in items) for source, items in by_problem.items()}
        problem_means = {source: c / n for source, c in outcomes.items()}
        cell = dict(zip(("model_id", "profile", "benchmark", "budget"), key))
        cell.update(
            family=selected[0]["family"],
            staleness=selected[0]["staleness"],
            problems=len(by_problem),
            samples_per_problem=n,
            completions=len(selected),
            correct=sum(outcomes.values()),
            mean_accuracy=statistics.fmean(problem_means.values()),
            accuracy_ci95=_interval(list(problem_means.values()), bootstrap_samples, rng),
            pass_at_k={str(k): statistics.fmean(pass_at_k(n, c, k) for c in outcomes.values()) for k in (1, 4, 8, 16) if k <= n},
            mean_token_count=statistics.fmean(row["token_count"] for row in selected),
            truncation_rate=statistics.fmean(row["truncated"] for row in selected),
            terminal_syntax_rate=statistics.fmean(row["terminal_syntax"] for row in selected),
        )
        cells.append(cell)
        problem_results[key] = problem_means
        grouped_records[key] = {(row["source_id"], row["sample_index"]): row for row in selected}
    bases = defaultdict(list)
    for cell in cells:
        if cell["staleness"] is None:
            bases[(cell["family"], cell["profile"], cell["benchmark"], cell["budget"])].append(cell)
    paired = []
    unpaired = []
    for cell in cells:
        if cell["staleness"] is None:
            continue
        pair_key = (cell["family"], cell["profile"], cell["benchmark"], cell["budget"])
        candidates = bases[pair_key]
        description = {name: cell[name] for name in ("model_id", "family", "staleness", "profile", "benchmark", "budget")}
        if len(candidates) != 1:
            unpaired.append({**description, "reason": "missing_base" if not candidates else "ambiguous_base"})
            continue
        base = candidates[0]
        trained_key = tuple(cell[name] for name in ("model_id", "profile", "benchmark", "budget"))
        base_key = tuple(base[name] for name in ("model_id", "profile", "benchmark", "budget"))
        trained_rows, base_rows = grouped_records[trained_key], grouped_records[base_key]
        if trained_rows.keys() != base_rows.keys():
            unpaired.append({**description, "base_model_id": base["model_id"], "reason": "question_or_sample_mismatch"})
            continue
        mismatch = next((name for sample in sorted(trained_rows) for name in ("comparison_sha256", "prompt_sha256", "seed", "answers", "heldout") if (name in trained_rows[sample] or name in base_rows[sample]) and trained_rows[sample].get(name) != base_rows[sample].get(name)), None)
        if mismatch is not None:
            unpaired.append({**description, "base_model_id": base["model_id"], "reason": f"paired_{mismatch}_mismatch"})
            continue
        differences = [problem_results[trained_key][source] - problem_results[base_key][source] for source in sorted(problem_results[base_key])]
        interval = _interval(differences, bootstrap_samples, rng)
        paired.append({
            **description,
            "base_model_id": base["model_id"],
            "problems": len(differences),
            "samples_per_problem": cell["samples_per_problem"],
            "base_accuracy": base["mean_accuracy"],
            "trained_accuracy": cell["mean_accuracy"],
            "delta_pp": statistics.fmean(differences) * 100,
            "delta_ci95_pp": None if interval is None else [value * 100 for value in interval],
            "pass_at_k_delta_pp": {k: (value - base["pass_at_k"][k]) * 100 for k, value in cell["pass_at_k"].items()},
        })
    return {
        "schema_version": 1,
        "grader": grader_provenance(),
        "bootstrap": {"unit": "problem", "samples": bootstrap_samples, "seed": seed, "method": "percentile", "confidence": 0.95},
        "responses": len(records),
        "cells": cells,
        "paired_deltas": paired,
        "unpaired": unpaired,
        "interpretation": "Mean sampled accuracy estimates pass@1. pass@k is an oracle success estimate. Confidence intervals resample problems, not individual completions. Terminal syntax does not imply successful mathematical parsing. Grader failures and timeouts are included as zero by the frozen grader.",
    }


def render_markdown(summary):
    def label(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    def interval(value, scale=100):
        return "—" if value is None else f"{value[0] * scale:.2f} to {value[1] * scale:.2f}"
    lines = ["# Math benchmark sweep", "", "| Model | Profile | Benchmark | Output tokens | Questions × samples | Accuracy | 95% CI | Truncated | Terminal syntax |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for cell in summary["cells"]:
        lines.append(f"| {label(cell['model_id'])} | {label(cell['profile'])} | {label(cell['benchmark'])} | {cell['budget']} | {cell['problems']} × {cell['samples_per_problem']} | {cell['mean_accuracy'] * 100:.2f}% | {interval(cell['accuracy_ci95'])} | {cell['truncation_rate'] * 100:.2f}% | {cell['terminal_syntax_rate'] * 100:.2f}% |")
    lines.extend(["", "## Paired changes from the starting model", "", "| Model | Profile | Benchmark | Output tokens | Change (pp) | 95% paired CI (pp) |", "|---|---|---|---:|---:|---:|"])
    for pair in summary["paired_deltas"]:
        lines.append(f"| {label(pair['model_id'])} | {label(pair['profile'])} | {label(pair['benchmark'])} | {pair['budget']} | {pair['delta_pp']:+.2f} | {interval(pair['delta_ci95_pp'], 1)} |")
    if summary["unpaired"]:
        lines.extend(["", "## Comparisons without a matched starting-model cell", ""])
        lines.extend(f"- {label(item['model_id'])}, {label(item['profile'])}, {label(item['benchmark'])}, {item['budget']}: {item['reason']}." for item in summary["unpaired"])
    lines.extend(["", summary["interpretation"], ""])
    return "\n".join(lines)
