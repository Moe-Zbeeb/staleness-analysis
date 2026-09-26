import hashlib
import json
from pathlib import Path

from deepseek_study import DATASET_ROWS, DATASET_SHA256, MODEL_ID, MODEL_REVISION
from deepseek_study.checkpoints import atomic_write


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    import pyarrow.parquet as pq

    if sha256(path) != DATASET_SHA256:
        raise ValueError("Dataset differs from the locked cleaned DeepScaleR release")
    rows = pq.read_table(path).to_pylist()
    if len(rows) != DATASET_ROWS or len({row["id"] for row in rows}) != DATASET_ROWS:
        raise ValueError("Unexpected dataset row count or duplicate question IDs")
    for row in rows:
        if len(row["answers"]) != 1 or not isinstance(row["answers"][0], str):
            raise ValueError("The locked dataset must have exactly one reference component per question")
        if row["messages"] != [{"role": "user", "content": row["prompt"]}]:
            raise ValueError("Question text and stored user message disagree")
    return rows


def prepare_tokenizer(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise ValueError("Prepared model destination must be a new directory")
    destination.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and path.name != "tokenizer_config.json" and not path.name.startswith("."):
            (destination / path.name).symlink_to(path)
    metadata = json.loads((source / "tokenizer_config.json").read_text())
    metadata["tokenizer_class"] = "TokenizersBackend"
    atomic_write(
        destination / "tokenizer_config.json", (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode()
    )
    return tokenizer_parity(source, destination)


def tokenizer_parity(source, destination):
    from renderers import create_renderer
    from renderers.configs import DefaultRendererConfig
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    native = Tokenizer.from_file(str(Path(source) / "tokenizer.json"))
    original = PreTrainedTokenizerFast.from_pretrained(source, local_files_only=True)
    loaded = AutoTokenizer.from_pretrained(destination, local_files_only=True)
    renderer = create_renderer(loaded, DefaultRendererConfig())
    probes = ["Hello world.\nTest", "  1 + 2 = 3\n\n", "\\boxed{\\frac{1}{2}}", "</think>\nAnswer: 42", "π ≠ 0"]
    for text in probes:
        expected = native.encode(text, add_special_tokens=False).ids
        if loaded.encode(text, add_special_tokens=False) != expected:
            raise ValueError("Prepared tokenizer changed native token IDs")
        if loaded.decode(expected, skip_special_tokens=False) != native.decode(expected, skip_special_tokens=False):
            raise ValueError("Prepared tokenizer changed native decoding")
        messages = [{"role": "user", "content": text}]
        expected_prompt = original.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        actual = renderer.render_ids(messages, add_generation_prompt=True)
        if actual != expected_prompt or not loaded.decode(actual).endswith("<think>\n"):
            raise ValueError("Renderer changed the native DeepSeek thinking prompt")
    if (loaded.bos_token_id, loaded.eos_token_id) != (151646, 151643):
        raise ValueError("Unexpected DeepSeek BOS/EOS tokens")
    return {"tokenizer_class": type(loaded).__name__, "probes_passed": len(probes), "native_template": True}


def prepare(model_path, dataset_path, destination, manifest_path):
    model_path, destination = Path(model_path).resolve(), Path(destination).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    if (manifest["repo_id"], manifest["revision"]) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("Incorrect model manifest")
    files = []
    for record in manifest["files"]:
        path = model_path / record["name"]
        if path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Model file failed verification: {path.name}")
        files.append({**record, "mtime_ns": path.stat().st_mtime_ns})
    rows = read_rows(dataset_path)
    parity = prepare_tokenizer(model_path, destination)
    receipt = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "source": str(model_path),
        "dataset_sha256": DATASET_SHA256,
        "rows": len(rows),
        "files": files,
        "tokenizer_parity": parity,
        "prepared_tokenizer_config_sha256": sha256(destination / "tokenizer_config.json"),
    }
    atomic_write(destination / "study-assets.json", (json.dumps(receipt, indent=2) + "\n").encode())
    return receipt


def validate_prepared(study):
    directory = study.prepared_model_path.resolve()
    receipt = json.loads((directory / "study-assets.json").read_text())
    if receipt["source"] != str(study.model_path.resolve()) or receipt["revision"] != MODEL_REVISION:
        raise ValueError("Prepared model belongs to another source")
    for record in receipt["files"]:
        path = study.model_path / record["name"]
        if path.stat().st_size != record["size"] or path.stat().st_mtime_ns != record["mtime_ns"]:
            raise ValueError("Original model changed after verification; prepare a fresh model view")
        if record["name"] != "tokenizer_config.json" and (directory / record["name"]).resolve() != path.resolve():
            raise ValueError("Prepared model no longer references its verified source")
    if sha256(directory / "tokenizer_config.json") != receipt["prepared_tokenizer_config_sha256"]:
        raise ValueError("Prepared tokenizer metadata changed")
    tokenizer_parity(study.model_path, directory)
    from deepseek_study.data import load_manifest, prepared_rows

    rows = prepared_rows(
        study.dataset_path,
        study.data_manifest,
        study.prompt_instruction,
        study.reward_timeout_seconds,
        study.prompt_max_tokens,
    )
    manifest = load_manifest(study.data_manifest)
    longest = max(record["prompt_tokens"] for record in manifest["records"] if record["included"])
    return {
        "rows": len(rows),
        "excluded_rows": manifest["excluded_rows"],
        "longest_prompt_tokens": longest,
        "data_manifest_sha256": manifest["sha256"],
    }
