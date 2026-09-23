import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from . import core


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def normalized(text):
    value = unicodedata.normalize("NFKC", text).lower()
    value = re.sub(r"\\(?:left|right|mathrm|mathbf|text|operatorname|displaystyle|quad|qquad)", " ", value)
    value = value.translate(str.maketrans({"+": " plus ", "-": " minus ", "=": " equals ", "^": " power ", "/": " over ", "<": " less ", ">": " greater "}))
    return " ".join(re.findall(r"[^\W_]+", value))


def shingles(text):
    tokens = text.split()
    return {tuple(tokens[index:index + 5]) for index in range(len(tokens) - 4)} if len(tokens) >= 5 else ({tuple(tokens)} if tokens else set())


def row_key(row):
    return f"{row['benchmark']}:{row['source_id']}"


def normalize_aime(answer):
    match = re.fullmatch(r"(\d{1,3})\s*(?:°|\^\\circ|\^\{\\circ\})?", answer.strip())
    if match is None or not 0 <= int(match.group(1)) <= 999:
        raise ValueError(f"Noninteger AIME gold: {answer!r}")
    return str(int(match.group(1)))


def _answers(value):
    values = value if isinstance(value, list) else [value]
    if not values or any(isinstance(item, (dict, list, bool)) or item is None for item in values):
        raise ValueError("Answers must be a nonempty list of scalar values")
    if any(isinstance(item, float) and not math.isfinite(item) for item in values):
        raise ValueError("Answers must be finite")
    result = [str(int(item)) if isinstance(item, float) and item.is_integer() else str(item).strip() for item in values]
    if any(not item for item in result):
        raise ValueError("Empty gold answer")
    return result


def _prompt(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Prompt must be nonempty text")
    value = value.strip()
    if value.startswith(core.INSTRUCTION):
        value = value[len(core.INSTRUCTION):].strip()
    if not value or not normalized(value):
        raise ValueError("Prompt has no auditable text")
    return value


def _canonical_row(record, spec):
    fields = spec["fields"]
    identifier = record.get(fields["source_id"])
    if identifier is None or isinstance(identifier, (list, dict, bool)) or not str(identifier).strip():
        raise ValueError(f"Invalid source ID in {spec['id']}")
    prompt = _prompt(record.get(fields["prompt"]))
    source_answers = _answers(record.get(fields["answers"]))
    answer_kind = spec["answer_kind"]
    if answer_kind not in {"math", "aime_integer"}:
        raise ValueError(f"Unsupported answer kind: {answer_kind}")
    answers = [normalize_aime(answer) for answer in source_answers] if answer_kind == "aime_integer" else list(source_answers)
    result = {"benchmark": spec["id"], "source_id": str(identifier), "prompt": prompt, "answers": answers, "source_answers": source_answers, "heldout": True, "overlap_status": "clear", "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    result["row_key"] = row_key(result)
    return result


def _read_frozen(spec, root, override, label):
    path = Path(override) if override is not None else Path(root) / spec["path"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen {label} file: {path}; use the original cluster data file, do not reconstruct it")
    expected = spec["sha256"]
    actual = core.file_hash(path)
    if actual != expected:
        raise ValueError(f"Frozen {label} SHA256 mismatch: {path}")
    records = core.read_jsonl(path)
    if len(records) != spec["count"]:
        raise ValueError(f"Frozen {label} count mismatch: {len(records)} != {spec['count']}")
    return records, {"path": str(path.resolve()), "sha256": actual, "count": len(records)}


def _download_rows(spec):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as parquet

    if not core.pinned(spec["revision"]):
        raise ValueError(f"Unpinned dataset revision: {spec['id']}")
    records = []
    files = spec.get("files", [])
    if not files or len(files) != len(set(files)):
        raise ValueError(f"Missing or duplicate pinned parquet files: {spec['id']}")
    for filename in files:
        expected = spec.get("file_sha256", {}).get(filename)
        if not filename.endswith(".parquet") or not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"Missing parquet hash: {spec['id']}/{filename}")
        path = hf_hub_download(repo_id=spec["repo_id"], repo_type="dataset", revision=spec["revision"], filename=filename)
        if core.file_hash(path) != expected:
            raise ValueError(f"Pinned parquet SHA256 mismatch: {spec['id']}/{filename}")
        records.extend(parquet.read_table(path).to_pylist())
    return records


def _unique(records, label):
    identities = set()
    prompts = set()
    for row in records:
        identity = (row["benchmark"], row["source_id"])
        prompt = (row["benchmark"], canonical(row["prompt"]))
        if identity in identities or prompt in prompts:
            raise ValueError(f"Duplicate {label} row: {identity}")
        identities.add(identity)
        prompts.add(prompt)


def _index_training(records):
    result = []
    exact = defaultdict(list)
    normalized_index = defaultdict(list)
    inverted = defaultdict(set)
    short_index = defaultdict(set)
    for index, record in enumerate(records):
        if not isinstance(record, dict) or "source_id" not in record:
            raise ValueError("Training rows require source_id")
        source_id = record["source_id"]
        if source_id is None or isinstance(source_id, (dict, list, bool)) or not str(source_id).strip():
            raise ValueError("Invalid training source_id")
        prompt = _prompt(record.get("prompt"))
        _answers(record.get("answers"))
        value = normalized(prompt)
        grams = shingles(value)
        result.append({"benchmark": "dapo_train", "source_id": str(source_id), "prompt": prompt, "normalized": value, "shingles": grams})
        exact[canonical(prompt)].append(index)
        normalized_index[value].append(index)
        for gram in grams:
            inverted[gram].add(index)
        for token in set(value.split()):
            short_index[token].add(index)
    _unique(result, "training")
    return result, exact, normalized_index, inverted, short_index


def _matches(row, training, exact, normalized_index, inverted, short_index):
    value = normalized(row["prompt"])
    exact_indices = exact.get(canonical(row["prompt"]), [])
    normalized_indices = normalized_index.get(value, [])
    if exact_indices or normalized_indices:
        reason = "exact" if exact_indices else "normalized"
        return [{"training_source_id": training[index]["source_id"], "kind": reason, "jaccard": 1.0, "sequence_ratio": 1.0} for index in exact_indices or normalized_indices]
    grams = shingles(value)
    intersections = Counter(index for gram in grams for index in inverted.get(gram, ()))
    candidates = set(intersections)
    short = len(value.split()) < 12
    if short:
        candidates.update(index for token in set(value.split()) for index in short_index.get(token, ()))
    matches = []
    for index in sorted(candidates):
        other = training[index]
        other_text = other["normalized"]
        length_ratio = min(len(value), len(other_text)) / max(len(value), len(other_text))
        if length_ratio < 0.50:
            continue
        union = len(grams) + len(other["shingles"]) - intersections[index]
        jaccard = intersections[index] / union if union else 0.0
        contained = length_ratio >= 0.72 and (value in other_text or other_text in value)
        if not contained and jaccard < 0.10 and not (short and length_ratio >= 0.80):
            continue
        matcher = SequenceMatcher(None, value, other_text, autojunk=False)
        sequence = matcher.ratio() if contained or jaccard >= 0.65 or matcher.quick_ratio() >= 0.90 else 0.0
        if contained or jaccard >= 0.65 or sequence >= 0.90:
            matches.append({"training_source_id": other["source_id"], "kind": "near_duplicate", "jaccard": jaccard, "sequence_ratio": sequence, "contained": contained, "training_prompt_sha256": hashlib.sha256(other["prompt"].encode()).hexdigest()})
    return sorted(matches, key=lambda item: (-item["jaccard"], -item["sequence_ratio"], item["training_source_id"]))


def _audit(rows, training_records, decisions):
    training, exact, normalized_index, inverted, short_index = _index_training(training_records)
    decisions = {} if decisions is None else decisions
    if not isinstance(decisions, dict):
        raise ValueError("Overlap decisions must be a mapping")
    known = {row["row_key"] for row in rows}
    if set(decisions) - known:
        raise ValueError(f"Unknown overlap decision rows: {sorted(set(decisions) - known)}")
    audit_rows = []
    cross_benchmark = defaultdict(list)
    for row in rows:
        cross_benchmark[normalized(row["prompt"])].append(row["row_key"])
        matches = _matches(row, training, exact, normalized_index, inverted, short_index)
        exact_match = bool(matches) and matches[0]["kind"] in {"exact", "normalized"}
        if exact_match:
            row.update(heldout=False, overlap_status=f"excluded_{matches[0]['kind']}")
        elif matches:
            row.update(heldout=False, overlap_status="unresolved_near_duplicate")
        decision = decisions.get(row["row_key"])
        if decision is not None:
            if not isinstance(decision, dict) or decision.get("decision") not in {"keep", "exclude"} or not isinstance(decision.get("evidence"), str) or not decision["evidence"].strip():
                raise ValueError(f"Overlap decision requires keep/exclude and evidence: {row['row_key']}")
            if decision.get("prompt_sha256") != row["prompt_sha256"]:
                raise ValueError(f"Overlap decision prompt hash mismatch: {row['row_key']}")
            if exact_match and decision["decision"] == "keep":
                raise ValueError(f"Cannot retain a confirmed exact/normalized training overlap: {row['row_key']}")
            if decision["decision"] == "keep" and not matches:
                raise ValueError(f"Keep decision has no candidate to review: {row['row_key']}")
            if not exact_match:
                row.update(heldout=decision["decision"] == "keep", overlap_status="reviewed_keep" if decision["decision"] == "keep" else "excluded_reviewed")
        if matches or decision:
            audit_rows.append({"row_key": row["row_key"], "prompt_sha256": row["prompt_sha256"], "status": row["overlap_status"], "matches": matches, "decision": decision})
    return {"schema_version": 1, "method": {"exact": "NFKC lowercase collapsed whitespace", "normalized": "frozen math operator normalization with Unicode letters and digits preserved", "shingle_width": 5, "review_jaccard": 0.65, "review_sequence_ratio": 0.90, "sequence_candidate_jaccard": 0.10, "review_containment_min_length_ratio": 0.72, "short_prompt_token_threshold": 12, "automatic_fuzzy_exclusion": False}, "training_count": len(training), "counts": dict(Counter(row["overlap_status"] for row in rows)), "heldout_count": sum(row["heldout"] for row in rows), "unresolved_count": sum(row["overlap_status"] == "unresolved_near_duplicate" for row in rows), "rows": audit_rows, "cross_benchmark_duplicates": [keys for keys in cross_benchmark.values() if len(keys) > 1], "decisions_sha256": core.digest(decisions)}


def prepare_data(catalog, root=core.ROOT, retained_path=None, training_path=None, decisions=None):
    catalog = core.read_json(catalog) if isinstance(catalog, (str, Path)) else catalog
    if catalog.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark catalog version")
    specs = catalog["benchmarks"]
    if not specs or len({spec["id"] for spec in specs}) != len(specs):
        raise ValueError("Missing or duplicate benchmark specifications")
    retained, retained_source = _read_frozen(catalog["retained_source"], root, retained_path, "evaluation")
    training, training_source = _read_frozen(catalog["training_source"], root, training_path, "training")
    retained_specs = {spec["id"]: spec for spec in specs if spec["source"] == "retained"}
    if set(row.get("benchmark") for row in retained) != set(retained_specs):
        raise ValueError("Frozen evaluation benchmark IDs differ from catalog")
    rows = []
    sources = {"catalog_sha256": core.digest(catalog), "retained": retained_source, "training": training_source, "benchmarks": {}}
    for spec in specs:
        if type(spec["expected_count"]) is not int or spec["expected_count"] < 1 or not core.pinned(spec["revision"]):
            raise ValueError(f"Invalid expected count or revision: {spec['id']}")
        if spec["source"] == "retained":
            records = [row for row in retained if row["benchmark"] == spec["id"]]
        elif spec["source"] == "huggingface":
            records = _download_rows(spec)
        else:
            raise ValueError(f"Unknown dataset source: {spec['source']}")
        if len(records) != spec["expected_count"]:
            raise ValueError(f"Benchmark count mismatch: {spec['id']} has {len(records)}, expected {spec['expected_count']}")
        prepared = [_canonical_row(record, spec) for record in records]
        _unique(prepared, spec["id"])
        rows.extend(prepared)
        sources["benchmarks"][spec["id"]] = {key: spec.get(key) for key in ("repo_id", "revision", "split", "source", "license", "file_sha256")}
        sources["benchmarks"][spec["id"]].update(count=len(prepared), normalized_gold_count=sum(row["answers"] != row["source_answers"] for row in prepared))
    _unique(rows, "evaluation")
    overlap_audit = _audit(rows, training, decisions)
    sources["canonical_rows_sha256"] = core.digest(rows)
    return {"rows": rows, "overlap_audit": overlap_audit, "sources": sources}
