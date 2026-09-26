import os
import shutil
import uuid
from pathlib import Path

from .common import digest, file_hash, lock, read, tokenizer_identity, write
from .discovery import checkpoint_receipt


def validate_export(directory, checkpoint_id=None, hashes=True):
    directory = Path(directory)
    receipt = read(directory / "evaluation-ready.json")
    if receipt.get("format") != 1 or (checkpoint_id and receipt["checkpoint_id"] != checkpoint_id):
        raise ValueError("Wrong exported model identity")
    for name, record in receipt["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or path.stat().st_size != record["size"]:
            raise ValueError("Export file missing or changed")
        if hashes and file_hash(path) != record["sha256"]:
            raise ValueError("Export content checksum mismatch")
    return receipt


def export_model(source, model_source, destination, shard_bytes=1024**3, progress=lambda: None):
    """Read only app.model from PrimeRL DCP, one bounded shard at a time.

    HF Qwen2 names/shapes are validated on a meta model. Optimizer tensors and
    the historical rollout queue are never allocated or deserialized.
    """
    import torch
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM

    checkpoint, destination, model_source = Path(source["path"]), Path(destination), Path(model_source)
    checkpoint_id = digest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with lock(destination.with_suffix(".lock")):
        if destination.exists():
            return validate_export(destination, checkpoint_id)
        if checkpoint_receipt(checkpoint) != source:
            raise ValueError("Checkpoint changed before export")
        config = AutoConfig.from_pretrained(model_source, local_files_only=True, trust_remote_code=False)
        if config.model_type != "qwen2" or config.tie_word_embeddings:
            raise ValueError("Exporter is validated only for the study's untied HF Qwen2 architecture")
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, attn_implementation="eager", dtype=torch.bfloat16)
        expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
        del model
        reader = dcp.FileSystemReader(checkpoint / "trainer")
        metadata = reader.read_metadata()
        prefix = "app.model."
        actual = {key[len(prefix):]: value for key, value in metadata.state_dict_metadata.items()
                  if key.startswith(prefix)}
        if set(actual) != set(expected):
            raise ValueError(f"DCP model keys differ: missing={set(expected)-set(actual)}, extra={set(actual)-set(expected)}")
        for key, value in actual.items():
            if tuple(value.size) != expected[key] or value.properties.dtype not in {torch.float32, torch.bfloat16}:
                raise ValueError(f"DCP model shape/dtype differs: {key}")
        # A source can be very large: check file identity around the read without
        # hashing optimizer bytes. Every exported tensor is covered by output hashes.
        source_stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
                        for p in (checkpoint / "trainer").iterdir() if p.is_file()}
        groups, current, size = [], [], 0
        for key, value in sorted(actual.items()):
            nbytes = value.size.numel() * torch.empty((), dtype=value.properties.dtype).element_size()
            if current and size + nbytes > shard_bytes:
                groups.append(current)
                current, size = [], 0
            current.append(key)
            size += nbytes
        if current:
            groups.append(current)
        temporary = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
        temporary.mkdir()
        try:
            weight_map, total_size = {}, 0
            for index, keys in enumerate(groups, 1):
                tensors = {key: torch.empty(actual[key].size, dtype=actual[key].properties.dtype) for key in keys}
                state = {"app": {"model": tensors}}
                dcp.load(state, storage_reader=reader, no_dist=True)
                tensors = {key: value.to(torch.bfloat16).contiguous() for key, value in tensors.items()}
                del state
                if any(not torch.isfinite(value).all().item() for value in tensors.values()):
                    raise ValueError("Non-finite checkpoint weights")
                filename = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
                save_file(tensors, str(temporary / filename), metadata={"format": "pt"})
                total_size += sum(value.numel() * value.element_size() for value in tensors.values())
                weight_map.update({key: filename for key in keys})
                del tensors
                progress()
            write(temporary / "model.safetensors.index.json", {
                "metadata": {"total_size": total_size}, "weight_map": weight_map,
            })
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
                         "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"):
                path = model_source / name
                if path.is_file():
                    shutil.copyfile(path, temporary / name)
            after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
                     for p in (checkpoint / "trainer").iterdir() if p.is_file()}
            if after != source_stats or checkpoint_receipt(checkpoint) != source:
                raise ValueError("Checkpoint changed during export")
            receipt = {
                "format": 1, "checkpoint_id": checkpoint_id, "source": source, "dtype": "bfloat16",
                "model_tensors": len(weight_map), "model_bytes": total_size,
                "tokenizer": tokenizer_identity(temporary),
                "files": {p.name: {"size": p.stat().st_size, "sha256": file_hash(p)}
                          for p in temporary.iterdir() if p.is_file()},
            }
            for path in temporary.iterdir():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            write(temporary / "evaluation-ready.json", receipt)
            os.rename(temporary, destination)
            from deepseek_study.runtime.checkpoints import fsync_directory

            fsync_directory(destination.parent)
            return receipt
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def publish_baseline(model_source, destination):
    """Copy an immutable model-only baseline, with content identity for reuse."""
    model_source, destination = Path(model_source), Path(destination)
    with lock(destination.with_suffix(".lock")):
        if destination.exists():
            return validate_export(destination)
        config = read(model_source / "config.json")
        if config["model_type"] != "qwen2":
            raise ValueError("Expected Qwen2 baseline")
        index = read(model_source / "model.safetensors.index.json")
        names = set(index["weight_map"].values()) | {"model.safetensors.index.json", "config.json",
                                                   "tokenizer.json", "tokenizer_config.json"}
        for optional in ("generation_config.json", "special_tokens_map.json", "chat_template.jinja"):
            if (model_source / optional).is_file():
                names.add(optional)
        files = {name: {"size": (model_source / name).stat().st_size, "sha256": file_hash(model_source / name)}
                 for name in sorted(names)}
        checkpoint_id = digest(files)
        temp = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
        temp.mkdir(parents=True)
        try:
            for name in names:
                if Path(name).name != name:
                    raise ValueError("Invalid baseline weight path")
                shutil.copyfile(model_source / name, temp / name)
            receipt = {"format": 1, "checkpoint_id": checkpoint_id, "source": {"baseline": True},
                       "files": files, "tokenizer": tokenizer_identity(temp)}
            write(temp / "evaluation-ready.json", receipt)
            validate_export(temp)
            os.rename(temp, destination)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
        return receipt
