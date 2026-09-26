import math
import os
import shutil
import socket
import sys
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

from pydantic import Field

from deepseek_study.dataset.grading import GraderPool

from .common import digest, io_slot
from .export import validate_export
from .protocol import Protocol, StrictModel


class WorkerProfile(StrictModel):
    tensor_parallel: int = Field(default=1, ge=1, le=2)
    max_sequences: int = Field(default=2, ge=1, le=16)
    memory_fraction: float = Field(default=0.9, gt=0, lt=1)
    enforce_eager: bool = True
    require_a100_40gb: bool = True
    local_model_cache: bool = True


def preflight_gpu(profile):
    import torch

    if torch.cuda.device_count() != profile.tensor_parallel:
        raise ValueError("Visible GPU count must exactly equal evaluator tensor parallelism")
    devices = []
    for index in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(index)
        if profile.require_a100_40gb and ("A100" not in gpu.name or not 38 * 1024**3 <= gpu.total_memory <= 42 * 1024**3):
            raise ValueError(f"Expected allocated A100 40GB; got {gpu.name}, {gpu.total_memory} bytes")
        devices.append({"name": gpu.name, "bytes": gpu.total_memory})
    return devices


def render_questions(protocol, model):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    rendered = {}
    for question in protocol.questions:
        prompt = question.question + ("\n\n" + protocol.instruction if protocol.instruction else "")
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                      add_generation_prompt=True, return_dict=False)
        if len(ids) > protocol.prompt_tokens:
            raise ValueError(f"Evaluation prompt exceeds frozen limit (no silent exclusion): {question.id}")
        if not tok.decode(ids, skip_special_tokens=False).endswith("<think>\n"):
            raise ValueError("Evaluation prompt did not open the native reasoning channel")
        rendered[question.id] = ids
    return tok, rendered


def configure_inference_environment():
    from prime_rl.configs.inference import InferenceConfig
    from prime_rl.inference.server import setup_vllm_env
    from prime_rl.utils.process import DEFAULT_INFERENCE_ENV_VARS

    os.environ.update(DEFAULT_INFERENCE_ENV_VARS)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ["VLLM_ENFORCE_STRICT_TOOL_CALLING"] = "0"
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    config = InferenceConfig()
    setup_vllm_env(config)
    return config.to_namespace().additional_config


class VLLMEngine:
    def __init__(self, model, protocol, profile):
        additional_config = configure_inference_environment()
        if shutil.which("ninja") is None:
            raise RuntimeError("Pinned runtime's ninja executable is unavailable on PATH")
        from vllm import LLM

        self.protocol, self.profile = protocol, profile
        self.hardware = preflight_gpu(profile)
        self.llm = LLM(
            model=str(model), tokenizer=str(model), dtype="bfloat16", trust_remote_code=False,
            tensor_parallel_size=profile.tensor_parallel, gpu_memory_utilization=profile.memory_fraction,
            max_model_len=protocol.prompt_tokens + protocol.response_tokens,
            max_num_seqs=profile.max_sequences, enforce_eager=profile.enforce_eager,
            enable_prefix_caching=False, generation_config="vllm", seed=protocol.seed,
            additional_config=additional_config,
            worker_extension_cls="prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
        )

    def generate(self, requests):
        from vllm import SamplingParams

        params = [SamplingParams(
            temperature=self.protocol.temperature, top_p=self.protocol.top_p, top_k=-1, min_p=0.0,
            repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
            max_tokens=self.protocol.response_tokens, seed=request["seed"], n=1,
            ignore_eos=False, skip_special_tokens=False,
        ) for request in requests]
        results = self.llm.generate([{"prompt_token_ids": r["prompt"]} for r in requests], params, use_tqdm=False)
        if len(results) != len(requests):
            raise ValueError("Incomplete inference batch")
        output = []
        for request, result in zip(requests, results, strict=True):
            if list(result.prompt_token_ids) != request["prompt"] or len(result.outputs) != 1:
                raise ValueError("Inference changed prompt IDs or response count")
            generated = result.outputs[0]
            if generated.finish_reason not in {"stop", "length"}:
                raise ValueError("Incomplete inference response")
            output.append({"token_ids": list(generated.token_ids), "finish_reason": generated.finish_reason,
                           "stop_reason": generated.stop_reason})
        return output


def summarize(protocol, rows, question_ids=None):
    expected = {key for key, *_ in protocol.samples(question_ids)}
    if len(rows) != len(expected) or {r["sample_id"] for r in rows} != expected:
        raise ValueError("Incomplete or duplicate evaluation results")
    by_question = defaultdict(list)
    for row in rows:
        if type(row["grade"]["correct"]) is not bool:
            raise ValueError("Grader failures cannot be counted as wrong answers")
        by_question[row["question_id"]].append(row)
    questions = []
    for question_id, group in by_question.items():
        questions.append({
            "question_id": question_id, "benchmark": group[0]["benchmark"], "responses": len(group),
            "correct": sum(r["grade"]["correct"] for r in group),
            "output_tokens": sum(r["output_tokens"] for r in group),
            "finished_tokens": sum(r["output_tokens"] for r in group if not r["truncated"]),
            "finished_count": sum(not r["truncated"] for r in group),
            "truncated_count": sum(r["truncated"] for r in group),
            "missing_final_count": sum(r["grade"]["extraction_status"] != "boxed" for r in group),
            "unsupported_count": sum(r["grade"]["status"] == "unsupported" for r in group),
        })
    return metrics_from_questions(questions)


def metrics_from_questions(questions):
    by_benchmark = defaultdict(list)
    for question in questions:
        by_benchmark[question["benchmark"]].append(question)
    metrics = {}
    for benchmark, group in by_benchmark.items():
        scores = [row["correct"] / row["responses"] for row in group]
        accuracy = sum(scores) / len(scores)
        count = sum(r["responses"] for r in group)
        finished = sum(r["finished_count"] for r in group)
        metrics[benchmark] = {
            "accuracy": accuracy, "questions": len(scores), "responses": count,
            "question_standard_error": math.sqrt(sum((s - accuracy)**2 for s in scores) /
                                                  (len(scores) * (len(scores)-1))) if len(scores) > 1 else None,
            "mean_output_tokens": sum(r["output_tokens"] for r in group) / count,
            "mean_finished_output_tokens": sum(r["finished_tokens"] for r in group) / finished if finished else None,
            "truncated_count": sum(r["truncated_count"] for r in group),
            "missing_final_count": sum(r["missing_final_count"] for r in group),
            "unsupported_count": sum(r["unsupported_count"] for r in group),
        }
        for name in ("truncated", "missing_final", "unsupported"):
            metrics[benchmark][name + "_rate"] = metrics[benchmark][name + "_count"] / count
    return {"metric": "mean per-question sampled pass@1", "benchmarks": metrics,
            "macro_accuracy": sum(m["accuracy"] for m in metrics.values()) / len(metrics), "question_metrics": questions}


async def _evaluate(queue, claim, profile, engine_factory, validate_backend, model, verified_export=None):
    spec = claim["spec"]
    protocol = Protocol.load(queue.root / "protocols" / f"{spec['protocol']}.json")
    if protocol.identity != spec["protocol"]:
        raise ValueError("Protocol content changed")
    if verified_export is None:
        with io_slot(queue.root):
            verified_export = validate_export(model, spec["checkpoint_id"])
    export = verified_export
    protocol.validate_runtime(model, backend=validate_backend)
    tok, prompts = render_questions(protocol, model)
    question_ids = spec.get("question_ids")
    if question_ids is not None:
        shards = list(protocol.shards())
        if spec["shards_total"] != len(shards) or shards[spec["shard"]] != question_ids:
            raise ValueError("Task question shard differs from frozen protocol")
    samples = list(protocol.samples(question_ids))
    engine = None
    command = [sys.executable, "-m", "deepseek_study.evaluation.grader_worker"]
    async with GraderPool(workers=1, timeout=12, retries=1, command=command) as grader:
        for start in range(0, len(samples), profile.max_sequences):
            batch = samples[start:start + profile.max_sequences]
            missing = [s for s in batch if not queue.sample_path(claim["id"], s[0], "raw").exists()]
            if missing:
                if engine is None:
                    with io_slot(queue.root):
                        engine = engine_factory(model, protocol, profile)
                started = time.time()
                outputs = engine.generate([{"prompt": prompts[q.id], "seed": seed} for _, q, _, seed in missing])
                if len(outputs) != len(missing):
                    raise ValueError("Incomplete inference batch")
                for (key, question, repeat, seed), output in zip(missing, outputs, strict=True):
                    tokens = output["token_ids"]
                    if not 0 < len(tokens) <= protocol.response_tokens or output["finish_reason"] not in {"stop", "length"}:
                        raise ValueError("Invalid generated response length/finish reason")
                    raw = {
                        "sample_id": key, "question_id": question.id, "benchmark": question.benchmark,
                        "repeat": repeat, "seed": seed, "protocol": protocol.identity,
                        "checkpoint_id": spec["checkpoint_id"], "prompt_token_ids": prompts[question.id],
                        "prompt_hash": digest(prompts[question.id]), "output_token_ids": tokens,
                        "raw": tok.decode(tokens, skip_special_tokens=False), "output_tokens": len(tokens),
                        "finish_reason": output["finish_reason"], "stop_reason": output.get("stop_reason"),
                        "truncated": output["finish_reason"] == "length", "batch_seconds": time.time() - started,
                        "worker": {"host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID"),
                                   "profile": profile.model_dump(), "hardware": engine.hardware,
                                   "versions": {name: version(name) for name in protocol.backend_versions}
                                   if validate_backend else {"test_backend": True}},
                    }
                    queue.save_sample(claim, key, "raw", raw)
            for key, question, repeat, seed in batch:
                raw = queue.read_sample(claim["id"], key, "raw")
                if (raw["protocol"], raw["checkpoint_id"], raw["seed"], raw["prompt_hash"], raw["sample_id"],
                    raw["question_id"], raw["benchmark"], raw["repeat"]) != (
                    protocol.identity, spec["checkpoint_id"], seed, digest(prompts[question.id]), key,
                    question.id, question.benchmark, repeat,
                ):
                    raise ValueError("Saved rollout provenance mismatch")
                grade_path = queue.sample_path(claim["id"], key, "grade")
                if not grade_path.exists():
                    response = await grader.call({"payload": {
                        "raw": raw["raw"], "gold": question.gold, "meta": question.meta,
                        "benchmark": question.benchmark, "question": question.question,
                    }})
                    queue.save_sample(claim, key, "grade", {"raw_sha256": digest(raw), "grade": response["result"]})
                if queue.read_sample(claim["id"], key, "grade")["raw_sha256"] != digest(raw):
                    raise ValueError("Grading record refers to a different response")
    rows = []
    for key, *_ in samples:
        raw = queue.read_sample(claim["id"], key, "raw")
        rows.append({**raw, "grade": queue.read_sample(claim["id"], key, "grade")["grade"]})
    metrics = summarize(protocol, rows, question_ids)
    result = {"spec": spec, "protocol": protocol.model_dump(), "export_sha256": digest(export),
              "predictions_sha256": digest(rows), "metrics": metrics}
    # Sample records are the durable source. Only the current lease may mark this
    # checkpoint complete; dashboards consume that receipt, never partial scores.
    queue.finish(claim, result)
    return result


async def evaluate(queue, claim, profile, engine_factory=VLLMEngine, validate_backend=True):
    if validate_backend and profile.local_model_cache:
        from .cache import staged_model

        with staged_model(queue.root, claim["spec"]["model"], claim["spec"]["checkpoint_id"]) as (model, receipt):
            return await _evaluate(queue, claim, profile, engine_factory, validate_backend, model, receipt)
    return await _evaluate(queue, claim, profile, engine_factory, validate_backend, claim["spec"]["model"])
