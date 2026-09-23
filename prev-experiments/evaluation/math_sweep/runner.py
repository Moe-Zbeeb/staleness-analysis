import fcntl
import importlib.metadata
import json
import math
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import check_context, digest, file_hash, pinned, read_json, seed_for, validate_records, write_json


REQUIRED_PACKAGES = {"torch", "vllm", "transformers", "verifiers", "math-verify", "huggingface-hub"}
ENGINE_KEYS = {"dtype", "tensor_parallel_size", "gpu_memory_utilization", "max_num_seqs", "max_num_batched_tokens", "enforce_eager", "enable_chunked_prefill", "enable_prefix_caching", "generation_config"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _strict_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError(f"Torn JSONL record in {path}; preserve the file and repair it explicitly before resuming")
    result = []
    for index, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank JSONL record at {path}:{index}")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid JSONL record at {path}:{index}; explicit repair required") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record must be an object at {path}:{index}")
        result.append(value)
    return result


def _verify_preparation(prepared_dir):
    prepared_dir = Path(prepared_dir)
    preparation = read_json(prepared_dir / "preparation.json")
    if preparation.get("schema_version") != 1 or not {"cell.json", "prompts.jsonl"} <= preparation.get("files", {}).keys():
        raise ValueError("Incomplete preparation manifest")
    for name, expected in preparation["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Preparation file paths must stay within the prepared directory")
        if file_hash(prepared_dir / relative) != expected:
            raise ValueError(f"Prepared input hash mismatch: {name}")
    cell = read_json(prepared_dir / "cell.json")
    if cell["cell_id"] != digest({key: value for key, value in cell.items() if key != "cell_id"})[:24]:
        raise ValueError("Prepared cell identity mismatch")
    if cell.get("input_files") != preparation.get("input_files") or not preparation.get("input_files"):
        raise ValueError("Prepared input manifests disagree")
    for name, expected in preparation["input_files"].items():
        if not Path(name).is_absolute() or file_hash(name) != expected:
            raise ValueError(f"Frozen input hash mismatch: {name}")
    versions = preparation["versions"]
    if cell["runtime"]["versions"] != versions or not REQUIRED_PACKAGES <= versions.keys():
        raise ValueError("Incomplete or inconsistent runtime versions")
    actual = {name: importlib.metadata.version(name) for name in versions}
    if actual != versions:
        raise ValueError(f"Runtime version mismatch: {[(name, actual[name], versions[name]) for name in versions if actual[name] != versions[name]]}")
    if cell["availability"] != "pinned" or not pinned(cell["model"]["revision"]):
        raise ValueError("Evaluation requires a pinned model revision")
    if cell["profile_config"]["context_kind"] != "native":
        raise NotImplementedError("Context extension requires the separate validated prime-workspace/experiments/aime-15b-staleness-length-v6 workflow and its rotary-cache, long-prefill and native-boundary gates; this runner supports native context only")
    if os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN"):
        raise ValueError("VLLM_ALLOW_LONG_MAX_MODEL_LEN must be unset for native-context evaluation")
    model_path = Path(cell["runtime"]["model_path"])
    config_path = model_path / "config.json"
    if str(config_path.resolve()) not in preparation["input_files"]:
        raise ValueError("Model config is missing from the frozen inputs")
    weights = list(model_path.glob("*.safetensors"))
    if not weights or any(str(item.resolve()) not in preparation["input_files"] for item in weights):
        raise ValueError("Model weights are missing from the frozen inputs")
    tokenizer_path = Path(cell["runtime"]["tokenizer_path"])
    tokenizer_files = [item for item in tokenizer_path.iterdir() if item.is_file() and (item.suffix in (".json", ".jinja") or item.name in ("vocab.txt", "merges.txt", "tokenizer.model", "spiece.model"))]
    if not tokenizer_files or any(str(item.resolve()) not in preparation["input_files"] for item in tokenizer_files):
        raise ValueError("Tokenizer files are missing from the frozen inputs")
    config = read_json(config_path)
    if not 0 < cell["context"] <= config["max_position_embeddings"]:
        raise ValueError("Engine context exceeds native model context")
    if cell["context"] > cell["family_config"]["native_context"]:
        raise ValueError("Engine context exceeds the family's declared native context")
    if set(cell["stop_token_ids"]) != {151643, 151645} or len(cell["stop_token_ids"]) != 2:
        raise ValueError("Both explicit Qwen EOS IDs are required")
    configured_eos = config.get("eos_token_id")
    if configured_eos is not None and not set(configured_eos if isinstance(configured_eos, list) else [configured_eos]) <= set(cell["stop_token_ids"]):
        raise ValueError("Model config contains an unexpected EOS ID")
    engine = cell["engine"]
    if set(engine) != ENGINE_KEYS or engine["generation_config"] != "vllm":
        raise ValueError("Unsupported or incomplete frozen engine settings")
    if engine["enable_prefix_caching"] or not engine["enforce_eager"] or engine["dtype"] != "bfloat16":
        raise ValueError("Unvalidated native engine settings")
    for name in ("tensor_parallel_size", "max_num_seqs", "max_num_batched_tokens"):
        if type(engine[name]) is not int or engine[name] < 1:
            raise ValueError(f"Invalid engine setting: {name}")
    profile = cell["profile_config"]
    if cell["samples"] != profile["samples"] or type(cell["samples"]) is not int or cell["samples"] < 1:
        raise ValueError("Sampling count mismatch")
    if profile["mode"] not in ("thinking", "nonthinking") or (profile["mode"] == "thinking" and cell["family"] != "qwen3-14b"):
        raise ValueError("Unsupported reasoning mode")
    rows = _strict_jsonl(prepared_dir / "prompts.jsonl")
    if len(rows) != cell["expected_problems"] or len(rows) * cell["samples"] != cell["expected_responses"]:
        raise ValueError("Prepared response counts disagree")
    if len({str(row["source_id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate prepared prompt identity")
    for row in rows:
        tokens = row["prompt_token_ids"]
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("Invalid prompt token IDs")
        if row["benchmark"] != cell["benchmark"] or row["prompt_sha256"] != digest(tokens):
            raise ValueError("Prepared prompt identity mismatch")
        if type(row["heldout"]) is not bool or not row["answers"] or not all(isinstance(answer, str) and answer.strip() for answer in row["answers"]):
            raise ValueError("Missing held-out decision or valid gold answers")
        check_context(tokens, cell["budget"], cell["context"])
    return cell, rows, preparation, file_hash(prepared_dir / "preparation.json")


def _allocation(cell):
    job_id = os.environ.get("SLURM_JOB_ID", "")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not job_id.isdigit() or not all(devices) or len(devices) != len(set(devices)) or any(device.strip() != device or device == "-1" for device in devices):
        raise ValueError("A Slurm job and explicit distinct CUDA_VISIBLE_DEVICES allocation are required")
    if len(devices) < cell["engine"]["tensor_parallel_size"]:
        raise ValueError("Insufficient visible GPUs for the frozen tensor-parallel layout")
    return {"job_id": job_id, "host": socket.gethostname(), "cuda_visible_devices": devices, "visible_gpu_count": len(devices), "engine_gpu_count": cell["engine"]["tensor_parallel_size"]}


def _load_engine(cell, allocation):
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available() or torch.cuda.device_count() != allocation["visible_gpu_count"]:
        raise ValueError("CUDA visibility does not match the Slurm allocation")
    hardware = cell["hardware"]
    devices = []
    for index in range(cell["engine"]["tensor_parallel_size"]):
        properties = torch.cuda.get_device_properties(index)
        if hardware["gpu_name_contains"] not in properties.name or properties.total_memory / 1024**3 < hardware["minimum_memory_gib"]:
            raise ValueError(f"GPU {index} does not meet the frozen hardware profile: {properties.name}, {properties.total_memory} bytes")
        devices.append({"index": index, "name": properties.name, "total_memory_bytes": properties.total_memory})
    kwargs = {**cell["engine"], "model": cell["runtime"]["model_path"], "tokenizer": cell["runtime"]["tokenizer_path"], "max_model_len": cell["context"], "seed": cell["seed"], "trust_remote_code": False}
    llm = LLM(**kwargs)
    effective = llm.llm_engine.model_config
    source_config = read_json(Path(cell["runtime"]["model_path"]) / "config.json")
    if effective.max_model_len != cell["context"] or effective.hf_config.max_position_embeddings != source_config["max_position_embeddings"]:
        raise ValueError("Effective vLLM context differs from the validated native context")
    tokenizer = llm.get_tokenizer()
    if {tokenizer.convert_tokens_to_ids(value) for value in ("<|endoftext|>", "<|im_end|>")} != set(cell["stop_token_ids"]):
        raise ValueError("Runtime tokenizer EOS identity mismatch")
    return llm, tokenizer, SamplingParams, {"engine": kwargs, "effective_hf_config": effective.hf_config.to_dict(), "gpus": devices, "torch_cuda_version": torch.version.cuda}


def _params(factory, cell, seed, **overrides):
    profile = cell["profile_config"]
    settings = {"n": 1, "temperature": profile["temperature"], "top_p": profile["top_p"], "top_k": profile["top_k"], "min_p": 0.0, "max_tokens": cell["budget"], "seed": seed, "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "stop_token_ids": cell["stop_token_ids"], "ignore_eos": False, "skip_special_tokens": False, "spaces_between_special_tokens": False}
    return factory(**{**settings, **overrides})


def _one_output(response, prompt_ids):
    if response.prompt_token_ids != prompt_ids or len(response.outputs) != 1 or not response.finished:
        raise ValueError("Generation changed the prompt, returned multiple completions, or did not finish")
    reply = response.outputs[0]
    if reply.finish_reason not in ("stop", "length"):
        raise ValueError(f"Generation failed: {reply.finish_reason}")
    return reply


def _probe_engine(llm, factory, cell, rows):
    prompt_ids = rows[0]["prompt_token_ids"]
    probes = {}
    for eos in cell["stop_token_ids"]:
        response = llm.generate([{"prompt_token_ids": prompt_ids}], _params(factory, cell, cell["seed"], temperature=0.0, max_tokens=min(8, cell["budget"]), logit_bias={eos: 100}), use_tqdm=False)
        if len(response) != 1:
            raise ValueError("EOS probe response count mismatch")
        reply = _one_output(response[0], prompt_ids)
        ids = list(reply.token_ids)
        if reply.finish_reason != "stop" or len(ids) > 1 or (reply.stop_reason != eos and ids != [eos]) or (ids and ids != [eos]):
            raise ValueError(f"Forced EOS stop gate failed for {eos}")
        probes[f"stop_{eos}"] = {"token_ids": ids, "finish_reason": reply.finish_reason, "stop_reason": reply.stop_reason}
    response = llm.generate([{"prompt_token_ids": prompt_ids}], _params(factory, cell, cell["seed"], temperature=0.0, max_tokens=min(32, cell["budget"]), logprobs=1), use_tqdm=False)
    if len(response) != 1:
        raise ValueError("Finite probe response count mismatch")
    reply = _one_output(response[0], prompt_ids)
    ids = list(reply.token_ids)
    if not ids or not reply.logprobs or len(ids) != len(reply.logprobs) or len(ids) > min(32, cell["budget"]):
        raise ValueError("Finite normal probe did not return aligned tokens and log probabilities")
    values = [probabilities[token].logprob for token, probabilities in zip(ids, reply.logprobs)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Finite normal decoding gate failed")
    probes["finite_normal"] = {"token_ids": ids, "selected_logprobs": values, "finish_reason": reply.finish_reason, "stop_reason": reply.stop_reason}
    return probes


def _validate_saved(records, rows, cell, preparation_sha256, complete=False):
    seen = validate_records(records, rows, cell, complete=complete)
    by_id = {str(row["source_id"]): row for row in rows}
    for record in records:
        row = by_id[str(record["source_id"])]
        expected = {key: cell[key] for key in ("model_id", "family", "staleness", "checkpoint_step", "profile", "benchmark", "budget", "comparison_sha256")}
        expected.update(preparation_sha256=preparation_sha256, heldout=row["heldout"], source_answers=row["source_answers"], mode=cell["profile_config"]["mode"], truncated=record["finish_reason"] == "length")
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("Saved response provenance mismatch")
        if any(type(record.get(key)) is not bool for key in ("correct", "terminal_syntax", "unfinished_thinking", "eos_seen")):
            raise ValueError("Saved response is missing validated scoring fields")
    return seen


def _record(reply, tokenizer, row, sample, cell, preparation_sha256):
    from .scoring import decode_completion, grade

    token_ids = list(reply.token_ids)
    decoded = decode_completion(token_ids, tokenizer, cell["stop_token_ids"], thinking_open=cell["profile_config"]["mode"] == "thinking")
    scored = grade(decoded["text"], row["answers"], unfinished_thinking=decoded["unfinished_thinking"])
    value = {key: cell[key] for key in ("cell_id", "model_id", "family", "staleness", "checkpoint_step", "profile", "benchmark", "budget", "comparison_sha256")}
    value.update(source_id=str(row["source_id"]), sample_index=sample, seed=seed_for(cell["benchmark"], row["source_id"], sample, cell["seed"]), prompt_sha256=row["prompt_sha256"], prompt_tokens=len(row["prompt_token_ids"]), answers=row["answers"], source_answers=row["source_answers"], heldout=row["heldout"], overlap_status=row.get("overlap_status"), mode=cell["profile_config"]["mode"], preparation_sha256=preparation_sha256, token_ids=token_ids, token_count=len(token_ids), engine_text=reply.text, finish_reason=reply.finish_reason, stop_reason=reply.stop_reason, truncated=reply.finish_reason == "length", **decoded, **scored)
    return value


def run_cell(prepared_dir: Path, output_root: Path):
    cell, rows, preparation, preparation_sha256 = _verify_preparation(prepared_dir)
    allocation = _allocation(cell)
    output = Path(output_root) / cell["cell_id"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Cell {cell['cell_id']} is already running") from error
        records_path = output / "records.jsonl"
        receipt_path = output / "receipt.json"
        records = _strict_jsonl(records_path)
        seen = _validate_saved(records, rows, cell, preparation_sha256)
        previous = read_json(receipt_path) if receipt_path.exists() else None
        if previous and (previous.get("preparation_sha256") != preparation_sha256 or previous.get("cell_id") != cell["cell_id"]):
            raise ValueError("Receipt belongs to a different preparation")
        if records and not previous:
            raise ValueError("Saved records have no preparation receipt")
        if previous and previous["status"] == "complete":
            _validate_saved(records, rows, cell, preparation_sha256, complete=True)
            if previous["records_sha256"] != file_hash(records_path) or previous["probes_sha256"] != file_hash(output / "probes.json"):
                raise ValueError("Completed result hash mismatch")
            return previous
        attempts = list(previous["attempts"]) if previous else []
        if attempts and attempts[-1]["status"] == "running":
            attempts[-1]["status"] = "interrupted"
            attempts[-1]["timing_complete"] = False
        attempt = {"index": len(attempts) + 1, "started_at": _now(), "status": "running", "allocation": allocation, "elapsed_seconds": 0.0, "gpu_hours": 0.0, "timing_complete": True}
        attempts.append(attempt)
        receipt = {"schema_version": 1, "status": "running", "cell_id": cell["cell_id"], "comparison_sha256": cell["comparison_sha256"], "preparation_sha256": preparation_sha256, "runtime_versions": preparation["versions"], "allocation": allocation, "expected_responses": cell["expected_responses"], "attempts": attempts, "timing_scope": "Engine initialization, decoding gates, generation and scoring after input verification; excludes scheduler wait and input hashing"}
        records_path.touch(exist_ok=True)
        started = time.monotonic()

        def save(status):
            attempt["status"] = status
            attempt["elapsed_seconds"] = time.monotonic() - started
            attempt["gpu_hours"] = attempt["elapsed_seconds"] * allocation["engine_gpu_count"] / 3600
            if status != "running":
                attempt["ended_at"] = _now()
            receipt.update(status=status, responses=len(records), records_sha256=file_hash(records_path), elapsed_seconds=sum(item["elapsed_seconds"] for item in attempts), gpu_hours=sum(item["gpu_hours"] for item in attempts), timing_complete=all(item["timing_complete"] for item in attempts), updated_at=_now())
            write_json(receipt_path, receipt)

        save("running")
        try:
            llm, tokenizer, factory, engine_info = _load_engine(cell, allocation)
            probes = {"schema_version": 1, "status": "passed", "cell_id": cell["cell_id"], "preparation_sha256": preparation_sha256, "allocation": allocation, **engine_info, "gates": _probe_engine(llm, factory, cell, rows)}
            write_json(output / f"probes-attempt-{attempt['index']}.json", probes)
            write_json(output / "probes.json", probes)
            receipt["probes_sha256"] = file_hash(output / "probes.json")
            attempt["probes_sha256"] = receipt["probes_sha256"]
            attempt["probes_file"] = f"probes-attempt-{attempt['index']}.json"
            jobs = [(row, sample) for sample in range(cell["samples"]) for row in rows if (str(row["source_id"]), sample) not in seen]
            batch_size = cell["engine"]["max_num_seqs"]
            with records_path.open("a", encoding="utf-8") as stream:
                for offset in range(0, len(jobs), batch_size):
                    batch = jobs[offset:offset + batch_size]
                    prompts = [{"prompt_token_ids": row["prompt_token_ids"]} for row, sample in batch]
                    params = [_params(factory, cell, seed_for(cell["benchmark"], row["source_id"], sample, cell["seed"])) for row, sample in batch]
                    responses = llm.generate(prompts, params, use_tqdm=False)
                    if len(responses) != len(batch):
                        raise ValueError("Generation response count mismatch")
                    for response, (row, sample) in zip(responses, batch):
                        reply = _one_output(response, row["prompt_token_ids"])
                        record = _record(reply, tokenizer, row, sample, cell, preparation_sha256)
                        _validate_saved([record], rows, cell, preparation_sha256)
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        records.append(record)
                    save("running")
            _validate_saved(records, rows, cell, preparation_sha256, complete=True)
            save("complete")
            return receipt
        except BaseException as error:
            receipt["error"] = {"type": type(error).__name__, "message": str(error)}
            save("failed")
            raise
