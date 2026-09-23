import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "evaluation"
INSTRUCTION = "Solve the following math problem. Explain your reasoning. End with either \\boxed{...} or a final line `Final answer: ...`.\n\n"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def pinned(revision):
    return isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None


def load_catalogs(config_dir=CONFIG):
    return tuple(read_json(Path(config_dir) / name) for name in ("models.json", "benchmarks.json", "profiles.json"))


def seed_for(benchmark, source_id, sample_index, seed=42):
    return (seed + int(digest([benchmark, str(source_id), sample_index])[:8], 16)) % (2**31)


def check_context(prompt_ids, budget, context):
    if not prompt_ids or budget < 1 or len(prompt_ids) + budget > context:
        raise ValueError(f"Context overflow: {len(prompt_ids)} prompt + {budget} output > {context}; no truncation allowed")


def make_plan(models, benchmarks, profiles, selected_models=None, selected_profiles=None):
    dataset_by_id = {item["id"]: item for item in benchmarks["benchmarks"]}
    known_models = {item["id"] for item in models["models"]}
    known_profiles = {item["id"] for item in profiles["profiles"]}
    if selected_models and set(selected_models) - known_models:
        raise ValueError("Unknown model selection")
    if selected_profiles and set(selected_profiles) - known_profiles:
        raise ValueError("Unknown profile selection")
    cells = []
    for model in models["models"]:
        if selected_models and model["id"] not in selected_models:
            continue
        family = models["families"][model["family"]]
        for profile in profiles["profiles"]:
            if selected_profiles is None and not profile["default"]:
                continue
            if selected_profiles is not None and profile["id"] not in selected_profiles:
                continue
            if profile.get("families") and model["family"] not in profile["families"]:
                continue
            for benchmark in profile["benchmarks"]:
                dataset = dataset_by_id[benchmark]
                for budget in profile.get("budgets_by_benchmark", {}).get(benchmark, profile["budgets"]):
                    value = {
                        "schema_version": 1,
                        "model_id": model["id"], "family": model["family"], "staleness": model["staleness"],
                        "checkpoint_step": model["checkpoint_step"], "model": model,
                        "family_config": family, "profile": profile["id"], "profile_config": profile,
                        "benchmark": benchmark, "dataset": dataset, "budget": budget,
                        "context": profile.get("context") or family["native_context"],
                        "samples": profile["samples"], "engine": profiles["engine"],
                        "hardware": profiles["hardware"],
                        "seed": profiles["seed"], "stop_token_ids": profiles["stop_token_ids"],
                        "instruction": INSTRUCTION,
                        "catalog_sha256": digest(benchmarks),
                        "expected_problems": dataset["expected_count"],
                        "expected_responses": dataset["expected_count"] * profile["samples"],
                        "availability": "pinned" if model["status"] == "pinned" and pinned(model["revision"]) else "awaiting_final",
                    }
                    value["cell_id"] = digest(value)[:24]
                    cells.append(value)
    if not cells:
        raise ValueError("Selection produced no cells")
    return {"schema_version": 1, "cells": cells, "summary": {
        "models": len({cell["model_id"] for cell in cells}), "cells": len(cells),
        "pinned_cells": sum(cell["availability"] == "pinned" for cell in cells),
        "responses_before_overlap_filter": sum(cell["expected_responses"] for cell in cells),
        "maximum_output_tokens": sum(cell["expected_responses"] * cell["budget"] for cell in cells),
    }}


def response_key(record):
    return str(record["source_id"]), record["sample_index"]


def validate_records(records, rows, cell, complete=False):
    identity_fields = ("cell_id", "model_id", "family", "profile", "benchmark", "staleness", "budget", "comparison_sha256", "checkpoint_step")
    if any(name not in cell for name in identity_fields):
        raise ValueError("Prepared cell is missing identity metadata")
    if len({str(row["source_id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate prepared question identity")
    expected = {(str(row["source_id"]), sample): row for row in rows for sample in range(cell["samples"])}
    seen = set()
    for record in records:
        if not isinstance(record.get("source_id"), str) or type(record.get("sample_index")) is not int:
            raise ValueError("Response source_id must be a string and sample_index an integer")
        key = response_key(record)
        if key in seen or key not in expected:
            raise ValueError(f"Duplicate or unexpected response: {key}")
        for name in identity_fields:
            if name not in record or type(record[name]) is not type(cell[name]) or record[name] != cell[name]:
                raise ValueError(f"Result identity mismatch: {name}")
        row = expected[key]
        if type(record.get("heldout")) is not bool or type(row.get("heldout")) is not bool or record["heldout"] != row["heldout"]:
            raise ValueError("Result held-out membership mismatch")
        if record["prompt_sha256"] != row["prompt_sha256"] or record["answers"] != row["answers"]:
            raise ValueError("Prompt or gold mismatch")
        if record["seed"] != seed_for(cell["benchmark"], key[0], key[1], cell["seed"]):
            raise ValueError("Sampling seed mismatch")
        if not isinstance(record.get("token_ids"), list) or any(type(token) is not int or token < 0 for token in record["token_ids"]):
            raise ValueError("Invalid completion token IDs")
        if type(record.get("token_count")) is not int or record["token_count"] != len(record["token_ids"]) or record["token_count"] > cell["budget"]:
            raise ValueError("Invalid token count")
        if record["finish_reason"] not in ("stop", "length"):
            raise ValueError("Failed generation must be retried before completion")
        if type(record.get("truncated")) is not bool or record["truncated"] != (record["finish_reason"] == "length"):
            raise ValueError("Truncation does not match finish reason")
        seen.add(key)
    if complete and seen != set(expected):
        raise ValueError(f"Incomplete cell: {len(seen)}/{len(expected)} responses")
    return seen
