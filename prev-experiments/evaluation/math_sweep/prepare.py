import importlib.metadata
import json
from pathlib import Path

from .core import ROOT, check_context, digest, file_hash, pinned, read_json, write_json


REQUIRED_VERSIONS = {
    "vllm": "0.26.0+cu129",
    "torch": "2.11.0+cu128",
    "transformers": "5.6.2",
    "verifiers": "0.3.1",
    "math-verify": "0.9.0",
    "latex2sympy2-extended": "1.11.0",
    "sympy": "1.14.0",
}


def runtime_versions():
    versions = {name: importlib.metadata.version(name) for name in [*REQUIRED_VERSIONS, "huggingface-hub", "tokenizers", "pyarrow"]}
    mismatches = {name: (versions[name], wanted) for name, wanted in REQUIRED_VERSIONS.items() if versions[name] != wanted}
    if mismatches:
        raise ValueError(f"Unvalidated runtime versions: {mismatches}")
    return versions


def verify_tokenizer(path, expected):
    hashes = {}
    for filename, wanted in expected.items():
        actual = file_hash(path / filename)
        if actual != wanted:
            raise ValueError(f"Frozen tokenizer mismatch: {path / filename}")
        hashes[str((path / filename).resolve())] = actual
    return hashes


def verify_model(path, model, family, training_sha256=None):
    config = read_json(path / "config.json")
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError("No safetensors model weights found")
    index = path / "model.safetensors.index.json"
    if index.exists():
        referenced = set(read_json(index)["weight_map"].values())
        if referenced != {item.name for item in files}:
            raise ValueError("Model weight index and shard files disagree")
    hashes = {str(item.resolve()): file_hash(item) for item in files}
    for item in path.glob("*.json"):
        hashes[str(item.resolve())] = file_hash(item)
    if model["staleness"] is not None:
        manifest_path = path / "export-manifest.json"
        exported = read_json(manifest_path)
        if exported.get("status") != "passed" or exported.get("step") != 1000 or model.get("checkpoint_step") != 1000:
            raise ValueError("Trained model lacks a validated final step-1000 export")
        if exported.get("base_model") != family["base_repo"] or exported.get("base_revision") != family["base_revision"]:
            raise ValueError("Export starting-model identity mismatch")
        required_files = {item.name for item in files} | {"config.json", "training-config.json"}
        if index.exists():
            required_files.add(index.name)
        if not required_files <= exported.get("files", {}).keys():
            raise ValueError("Export manifest does not cover every weight shard and training identity")
        if {name for name in exported["files"] if name.endswith(".safetensors")} != {item.name for item in files}:
            raise ValueError("Export manifest weight shards differ from the actual model")
        for name, info in exported["files"].items():
            if name in required_files:
                if file_hash(path / name) != info["sha256"]:
                    raise ValueError(f"Export file mismatch: {name}")
        training = read_json(path / "training-config.json")
        if training.get("max_off_policy_steps") != model["staleness"] or training.get("training_steps") != 1000:
            raise ValueError("Export staleness or training step mismatch")
        if training.get("base_model") != family["base_repo"] or training.get("base_revision") != family["base_revision"]:
            raise ValueError("Training starting-model identity mismatch")
        if training.get("training_manifest_sha256") != exported.get("training_manifest_sha256"):
            raise ValueError("Training provenance mismatch")
        if training_sha256 and exported.get("training_data_sha256") != training_sha256:
            raise ValueError("Overlap audit and model training data differ")
        validation = exported.get("validation", {})
        if not all(validation.get(key) for key in ("finite_tensors", "strict_hf_reload", "identical_cpu_probe_logits")):
            raise ValueError("Export integrity validation is incomplete")
    return config, hashes


def tokenize_rows(rows, tokenizer, cell):
    prepared = []
    for row in rows:
        if row["benchmark"] != cell["benchmark"]:
            continue
        if row.get("overlap_status") == "unresolved_near_duplicate":
            raise ValueError(f"Review training overlap before generation: {row['benchmark']}/{row['source_id']}")
        if not isinstance(row.get("heldout"), bool):
            raise ValueError("Missing overlap decision")
        kwargs = {"enable_thinking": True} if cell["profile_config"]["mode"] == "thinking" else {}
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": cell["instruction"] + row["prompt"]}],
            tokenize=True, add_generation_prompt=True, return_dict=False, **kwargs,
        )
        check_context(tokens, cell["budget"], cell["context"])
        rendered = tokenizer.decode(tokens, skip_special_tokens=False)
        if cell["family"] == "qwen3-14b":
            opened = rendered.rfind("<think>") > rendered.rfind("</think>")
            if opened != (cell["profile_config"]["mode"] == "thinking"):
                raise ValueError("Qwen3 prompt mode mismatch")
            if cell["profile_config"]["mode"] == "nonthinking" and "</think>" not in rendered:
                raise ValueError("Missing frozen Qwen3 empty thinking prefix")
        prepared.append({**row, "source_prompt_sha256": row.get("prompt_sha256"), "prompt_token_ids": tokens, "prompt_sha256": digest(tokens)})
    if len(prepared) != cell["expected_problems"]:
        raise ValueError("Prepared dataset membership does not match the frozen plan")
    if len({row["source_id"] for row in prepared}) != len(prepared):
        raise ValueError("Duplicate prompt identity")
    return sorted(prepared, key=lambda row: str(row["source_id"]))


def prepare_cell(cell, data_path, output_dir, tokenizer_root=ROOT):
    if cell["availability"] != "pinned" or not pinned(cell["model"]["revision"]):
        raise ValueError("This model is awaiting a verified final checkpoint and immutable revision")
    data_path = Path(data_path)
    bundle = read_json(data_path)
    receipt = read_json(data_path.with_suffix(".receipt.json"))
    if receipt["sha256"] != file_hash(data_path):
        raise ValueError("Data preparation receipt mismatch")
    if receipt["catalog_sha256"] != cell["catalog_sha256"]:
        raise ValueError("Data was prepared from a different dataset catalog")
    versions = runtime_versions()
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from .scoring import validate_grader

    grader_validation = validate_grader()
    model_path = Path(snapshot_download(cell["model"]["repo_id"], revision=cell["model"]["revision"], allow_patterns=["*.json", "*.safetensors", "*.jinja"]))
    model_config, hashes = verify_model(model_path, cell["model"], cell["family_config"], bundle["sources"]["training"]["sha256"])
    if cell["profile_config"]["context_kind"] == "native" and cell["context"] > model_config["max_position_embeddings"]:
        raise ValueError("Configured context exceeds the actual model context")
    if cell["profile_config"]["mode"] == "thinking":
        tokenizer_path = Path(snapshot_download(cell["family_config"]["base_repo"], revision=cell["family_config"]["base_revision"], allow_patterns=["tokenizer*", "*.jinja", "vocab.json", "merges.txt", "config.json"]))
        tokenizer_hashes = {str(item.resolve()): file_hash(item) for item in tokenizer_path.iterdir() if item.is_file() and (item.name.startswith("tokenizer") or item.suffix == ".jinja" or item.name in ("vocab.json", "merges.txt", "config.json"))}
    else:
        tokenizer_path = Path(tokenizer_root) / cell["family_config"]["tokenizer"]["path"]
        tokenizer_hashes = verify_tokenizer(tokenizer_path, cell["family_config"]["tokenizer"]["file_sha256"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    if {tokenizer.convert_tokens_to_ids(value) for value in ("<|endoftext|>", "<|im_end|>")} != set(cell["stop_token_ids"]):
        raise ValueError("EOS identity mismatch")
    rows = tokenize_rows(bundle["rows"], tokenizer, cell)
    input_files = {**hashes, **tokenizer_hashes, str(data_path.resolve()): file_hash(data_path)}
    code_hashes = {}
    for item in Path(__file__).parent.glob("*.py"):
        if item.name.startswith("test_"):
            continue
        input_files[str(item.resolve())] = file_hash(item)
        code_hashes[item.name] = file_hash(item)
    grader = ROOT / "packages/prime-rl-staleness/src/exact_math.py"
    input_files[str(grader)] = file_hash(grader)
    final = {**cell, "runtime": {"model_path": str(model_path), "tokenizer_path": str(tokenizer_path), "versions": versions},
             "data_sha256": file_hash(data_path), "input_files": input_files, "grader_validation": grader_validation}
    final["comparison_sha256"] = digest({
        "engine": cell["engine"], "hardware": cell["hardware"], "versions": versions, "profile": cell["profile_config"], "budget": cell["budget"],
        "tokenizer": sorted(tokenizer_hashes.values()), "rows": [(row["source_id"], row["prompt_sha256"], row["answers"], row["heldout"]) for row in rows],
        "grader": file_hash(grader), "stop_ids": cell["stop_token_ids"], "seed": cell["seed"],
        "implementation": code_hashes,
    })
    del final["cell_id"]
    final["cell_id"] = digest(final)[:24]
    output = Path(output_dir) / final["cell_id"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "preparation.json").exists():
        existing = read_json(output / "preparation.json")
        if all(file_hash(output / name) == sha for name, sha in existing["files"].items()):
            return output
        raise ValueError("Existing prepared cell has changed")
    write_json(output / "cell.json", final)
    (output / "prompts.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write_json(output / "preparation.json", {
        "schema_version": 1, "source_cell_id": cell["cell_id"],
        "files": {name: file_hash(output / name) for name in ("cell.json", "prompts.jsonl")},
        "input_files": input_files, "versions": versions,
        "overlap_audit_sha256": digest(bundle["overlap_audit"]),
    })
    return output
