import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

DAPO_REVISION = "31dd309567e3da778038cc87d868b6097a3ccf68"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
AIME24_REVISION = "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"
AIME25_REVISION = "a6ad95f611d72cf628a80b58bd0432ef6638f958"
AIME26_REVISION = "10b4e45b7a503075d4da8a0d57916a4f06ce6bd2"
INTELLECT_REVISION = "0c6d5c96f64bb8981220d8a280a8f998a8b1642f"
INSTRUCTION = "Solve the following math problem. Explain your reasoning. End with either \\boxed{...} or a final line `Final answer: ...`.\n\n"


def normalized(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    value = re.sub(
        r"\\(?:left|right|mathrm|mathbf|text|operatorname|displaystyle|quad|qquad)",
        " ",
        value,
    )
    value = value.translate(
        str.maketrans(
            {
                "+": " plus ",
                "-": " minus ",
                "=": " equals ",
                "^": " power ",
                "/": " over ",
                "<": " less ",
                ">": " greater ",
            }
        )
    )
    return " ".join(re.findall(r"[a-z0-9]+", value))


def canonical(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def shingles(text: str, width: int = 5) -> set[tuple[str, ...]]:
    tokens = text.split()
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {
        tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)
    }


def boxed_answer(text: str) -> str | None:
    marker = text.rfind("\\boxed")
    if marker < 0:
        return None
    start = text.find("{", marker)
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index].strip()
    return None


def answer_list(value) -> list[str]:
    values = value if isinstance(value, list) else [value]
    answers = []
    for item in values:
        if isinstance(item, float) and item.is_integer():
            answers.append(str(int(item)))
        else:
            answers.append(str(item).strip())
    return [answer for answer in answers if answer]


def read_jsonl(path: Path):
    with path.open() as source:
        for line in source:
            yield json.loads(line)


def append_eval(rows: list[dict], benchmark: str, prompt, answers, source_id) -> None:
    resolved = answer_list(answers)
    if prompt and resolved:
        rows.append(
            {
                "benchmark": benchmark,
                "prompt": str(prompt).strip(),
                "answers": resolved,
                "source_id": str(source_id),
            }
        )


def collect_eval(intellect_root: Path) -> list[dict]:
    rows = []
    math500 = load_dataset(
        "HuggingFaceH4/MATH-500",
        split="test",
        revision=MATH500_REVISION,
    )
    for row in math500:
        append_eval(rows, "math500", row["problem"], row["answer"], row["unique_id"])
    aime24 = load_dataset(
        "HuggingFaceH4/aime_2024",
        split="train",
        revision=AIME24_REVISION,
    )
    for row in aime24:
        append_eval(rows, "aime24", row["problem"], row["answer"], row["id"])
    for subset in ("AIME2025-I", "AIME2025-II"):
        aime25 = load_dataset(
            "opencompass/AIME2025",
            subset,
            split="test",
            revision=AIME25_REVISION,
        )
        for index, row in enumerate(aime25):
            append_eval(
                rows, "aime25", row["question"], row["answer"], f"{subset}-{index}"
            )
    aime26 = load_dataset(
        "MathArena/aime_2026",
        split="train",
        revision=AIME26_REVISION,
    )
    for row in aime26:
        append_eval(rows, "aime26", row["problem"], row["answer"], row["problem_idx"])
    amc_path = (
        intellect_root
        / "rl-and-evals/eval/data/AI-MO/aimo-validation-amc/aimo-validation-amc.jsonl"
    )
    for row in read_jsonl(amc_path):
        if "2023_AMC" in str(row.get("url", "")):
            append_eval(rows, "amc23", row["question"], row["answer"], row["url"])
    minerva_path = intellect_root / "rl-and-evals/eval/data/minerva_math/test.jsonl"
    for row in read_jsonl(minerva_path):
        answer = boxed_answer(row["solution"])
        if answer is not None:
            append_eval(rows, "minerva_math", row["problem"], answer, row["idx"])
    olympiad_path = intellect_root / "rl-and-evals/eval/data/olympiadbench/test.jsonl"
    for row in read_jsonl(olympiad_path):
        append_eval(
            rows, "olympiadbench", row["question"], row["final_answer"], row["id"]
        )
    return rows


def prompt_tokens(tokenizer, prompt: str) -> int:
    return len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": INSTRUCTION + prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
    )


def contamination_match(train_text: str, eval_rows: list[dict], exact, index):
    if train_text in exact:
        candidate = exact[train_text][0]
        return candidate, 1.0, 1.0
    train_shingles = shingles(train_text)
    counts = Counter()
    for shingle in train_shingles:
        for candidate in index.get(shingle, ()):
            counts[candidate] += 1
    minimum = max(2, int(len(train_shingles) * 0.25))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    for candidate, intersection in ranked:
        if intersection < minimum:
            break
        eval_text = eval_rows[candidate]["normalized"]
        eval_shingles = eval_rows[candidate]["shingles"]
        union = len(train_shingles | eval_shingles)
        jaccard = intersection / union if union else 0.0
        containment = min(len(train_text), len(eval_text)) / max(
            len(train_text), len(eval_text)
        )
        contained = (
            train_text in eval_text or eval_text in train_text
        ) and containment >= 0.72
        sequence = (
            SequenceMatcher(None, train_text, eval_text, autojunk=False).ratio()
            if jaccard >= 0.35
            else 0.0
        )
        if contained or jaccard >= 0.65 or sequence >= 0.90:
            return candidate, jaccard, sequence
    return None


def write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_hashes(path: Path) -> dict[str, str]:
    index_path = path / "model.safetensors.index.json"
    weights = (
        {"model.safetensors.index.json", *json.loads(index_path.read_text())["weight_map"].values()}
        if index_path.is_file()
        else {"model.safetensors"}
    )
    names = {
        "MODEL_INFO.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        *weights,
    }
    missing = sorted(name for name in names if not (path / name).is_file())
    if missing:
        raise ValueError(f"Missing model artifacts: {missing}")
    return {name: file_hash(path / name) for name in sorted(names)}


def directory_hashes(path: Path) -> dict[str, str]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    return {str(item.relative_to(path)): file_hash(item) for item in files}


def answer_key(value) -> tuple[str, ...]:
    return tuple(sorted(canonical(answer) for answer in answer_list(value)))


def verify_reference_data(reference: Path, generated: dict[str, list[dict]]) -> dict:
    manifest_path = reference / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verified = {}
    for name, values in generated.items():
        source = reference / name
        digest = file_hash(source)
        if digest != manifest["hashes"][name]:
            raise ValueError(f"Reference data does not match its manifest: {name}")
        source_rows = list(read_jsonl(source))
        semantic = lambda rows: [
            {key: value for key, value in row.items() if key != "prompt_tokens"}
            for row in rows
        ]
        if semantic(values) != semantic(source_rows):
            raise ValueError(f"The 1.5B rows or their order differ from the 7B experiment: {name}")
        verified[name] = digest
    return {"directory": str(reference), "manifest_sha256": file_hash(manifest_path), "hashes": verified}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer-output", type=Path, required=True)
    parser.add_argument("--intellect-root", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--observer-archive", type=Path, required=True)
    parser.add_argument("--reference-data", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exclusions = json.loads(args.exclusions.read_text())
    if exclusions.get("conflicting_duplicate_policy") != "drop_group":
        raise ValueError("Unsupported conflicting duplicate policy")
    audited_overlaps = exclusions["audited_overlap_source_ids"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.eos_token = "<|im_end|>"
    args.tokenizer_output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.tokenizer_output)
    eval_rows = collect_eval(args.intellect_root)
    eval_filtered = []
    eval_length_drops = Counter()
    for row in eval_rows:
        tokens = prompt_tokens(tokenizer, row["prompt"])
        limit = 2048 if row["benchmark"] in {"minerva_math", "olympiadbench"} else 1024
        if tokens > limit:
            eval_length_drops[row["benchmark"]] += 1
            continue
        row["prompt_tokens"] = tokens
        row["normalized"] = normalized(row["prompt"])
        row["shingles"] = shingles(row["normalized"])
        eval_filtered.append(row)
    eval_sources = {
        (row["benchmark"], str(row["source_id"])) for row in eval_filtered
    }
    invalid_exclusions = sorted(
        source_id
        for source_id, heldout in audited_overlaps.items()
        if (heldout["benchmark"], str(heldout["eval_source_id"]))
        not in eval_sources
    )
    if invalid_exclusions:
        raise ValueError(
            f"Audited overlap metadata does not identify held-out rows: {invalid_exclusions}"
        )
    exact = defaultdict(list)
    index = defaultdict(set)
    for candidate, row in enumerate(eval_filtered):
        exact[row["normalized"]].append(candidate)
        for shingle in row["shingles"]:
            index[shingle].add(candidate)
    dapo = load_dataset(
        "open-r1/DAPO-Math-17k-Processed",
        "all",
        split="train",
        revision=DAPO_REVISION,
    )
    grouped = defaultdict(list)
    for row in dapo:
        grouped[canonical(str(row["prompt"]).strip())].append(row)
    representatives = []
    conflict_records = []
    duplicate_count = 0
    for group in grouped.values():
        duplicate_count += len(group) - 1
        golds = {answer_key(row["solution"]) for row in group}
        if len(golds) > 1:
            conflict_records.append(
                {
                    "canonical_prompt_sha256": hashlib.sha256(
                        canonical(str(group[0]["prompt"]).strip()).encode()
                    ).hexdigest(),
                    "rows": sorted(
                        (
                            {
                                "source_id": str(row["extra_info"]["index"]),
                                "answers": answer_list(row["solution"]),
                            }
                            for row in group
                        ),
                        key=lambda row: row["source_id"],
                    ),
                }
            )
            continue
        representatives.append(
            min(group, key=lambda row: str(row["extra_info"]["index"]))
        )
    train_rows = []
    overlap_records = []
    automatic_overlap_count = 0
    seen_audited_overlaps = set()
    length_drop_count = 0
    for row in representatives:
        prompt = str(row["prompt"]).strip()
        source_id = str(row["extra_info"]["index"])
        key = normalized(prompt)
        if source_id in audited_overlaps:
            heldout = audited_overlaps[source_id]
            overlap_records.append(
                {
                    "train_source_id": source_id,
                    "benchmark": heldout["benchmark"],
                    "eval_source_id": heldout["eval_source_id"],
                    "reason": "audited_variant",
                }
            )
            seen_audited_overlaps.add(source_id)
            continue
        match = contamination_match(key, eval_filtered, exact, index)
        if match is not None:
            candidate, jaccard, sequence = match
            heldout = eval_filtered[candidate]
            automatic_overlap_count += 1
            overlap_records.append(
                {
                    "train_source_id": source_id,
                    "benchmark": heldout["benchmark"],
                    "eval_source_id": heldout["source_id"],
                    "jaccard": jaccard,
                    "sequence_ratio": sequence,
                    "reason": "automatic_similarity",
                }
            )
            continue
        tokens = prompt_tokens(tokenizer, prompt)
        if tokens > 1024:
            length_drop_count += 1
            continue
        train_rows.append(
            {
                "benchmark": "dapo_train",
                "prompt": prompt,
                "answers": answer_list(row["solution"]),
                "source_id": source_id,
                "prompt_tokens": tokens,
            }
        )
    if seen_audited_overlaps != set(audited_overlaps):
        missing = sorted(set(audited_overlaps) - seen_audited_overlaps)
        raise ValueError(f"Audited overlap IDs were not found: {missing}")
    random.Random(args.seed).shuffle(train_rows)
    for row in eval_filtered:
        row.pop("normalized")
        row.pop("shingles")
    reference_data = verify_reference_data(
        args.reference_data,
        {
            "train.jsonl": train_rows,
            "eval.jsonl": eval_filtered,
            "removed_overlaps.jsonl": overlap_records,
            "removed_conflicting_duplicates.jsonl": conflict_records,
        },
    )
    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "eval.jsonl"
    overlap_path = args.output_dir / "removed_overlaps.jsonl"
    conflict_path = args.output_dir / "removed_conflicting_duplicates.jsonl"
    train_hash = write_jsonl(train_path, train_rows)
    eval_hash = write_jsonl(eval_path, eval_filtered)
    overlap_hash = write_jsonl(overlap_path, overlap_records)
    conflict_hash = write_jsonl(conflict_path, conflict_records)
    critical_inputs = (
        "config/data_exclusions.json",
        "config/main.toml",
        "config/smoke.toml",
        "experiment.json",
        "observer/pyproject.toml",
        "observer/uv.lock",
        "python/exact_math.py",
        "python/ppo_loss.py",
        "scripts/opik_bridge.py",
        "scripts/prepare_data.py",
        "scripts/prepare_model.py",
        "scripts/prepare_job.sh",
        "scripts/health_probe.py",
        "scripts/setup_opik_job.sh",
        "scripts/smoke_job.sh",
        "scripts/train_job.sh",
        "scripts/validate_experiment.py",
    )
    input_hashes = {
        name: file_hash(args.experiment_root / name) for name in critical_inputs
    }
    manifest = {
        "seed": args.seed,
        "reference_data": reference_data,
        "dataset_revisions": {
            "dapo": DAPO_REVISION,
            "math500": MATH500_REVISION,
            "aime24": AIME24_REVISION,
            "aime25": AIME25_REVISION,
            "aime26": AIME26_REVISION,
            "intellect_math": INTELLECT_REVISION,
        },
        "dapo_loaded": len(dapo),
        "dapo_kept": len(train_rows),
        "dapo_duplicate_drops": duplicate_count,
        "dapo_conflicting_duplicate_groups": len(conflict_records),
        "dapo_conflicting_duplicate_rows": sum(
            len(record["rows"]) for record in conflict_records
        ),
        "dapo_overlap_drops": len(overlap_records),
        "dapo_automatic_overlap_drops": automatic_overlap_count,
        "dapo_audited_overlap_drops": len(seen_audited_overlaps),
        "dapo_length_drops": length_drop_count,
        "eval_counts": dict(
            sorted(Counter(row["benchmark"] for row in eval_filtered).items())
        ),
        "eval_length_drops": dict(sorted(eval_length_drops.items())),
        "prompt_token_limits": {
            "train": 1024,
            "standard_eval": 1024,
            "long_eval": 2048,
        },
        "hashes": {
            "train.jsonl": train_hash,
            "eval.jsonl": eval_hash,
            "removed_overlaps.jsonl": overlap_hash,
            "removed_conflicting_duplicates.jsonl": conflict_hash,
        },
        "input_hashes": input_hashes,
        "model": str(args.model),
        "model_hashes": model_hashes(args.model),
        "observer_archive": str(args.observer_archive),
        "observer_archive_sha256": file_hash(args.observer_archive),
        "tokenizer": str(args.tokenizer_output),
        "tokenizer_hashes": directory_hashes(args.tokenizer_output),
        "tokenizer_eos_token": tokenizer.eos_token,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
