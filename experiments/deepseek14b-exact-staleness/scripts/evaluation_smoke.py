"""Bounded CPU preparation and real 40GB GPU acceptance tests (not benchmark scores)."""
import argparse
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from deepseek_study import MODEL_REVISION
from deepseek_study.dataset.assets import prepare_tokenizer
from deepseek_study.evaluation.common import immutable_json, read, write
from deepseek_study.evaluation.cache import staged_model
from deepseek_study.evaluation.discovery import enqueue_evaluation
from deepseek_study.evaluation.export import publish_baseline
from deepseek_study.evaluation.protocol import Protocol, from_sweep
from deepseek_study.evaluation.queue import Queue
from deepseek_study.evaluation.runner import VLLMEngine, WorkerProfile, configure_inference_environment, render_questions
from deepseek_study.evaluation.service import initialize, work


def prepare(args):
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    info = read(args.model_record)
    if info["revision" if "revision" in info else "sha"] != MODEL_REVISION:
        raise ValueError("Smoke model revision differs from the training initialization")
    original = Path(info["path"])
    native = root / "native-tokenizer"
    if not native.exists():
        prepare_tokenizer(original, native)
    baseline = root / "baseline"
    if args.baseline and not baseline.exists():
        baseline.symlink_to(args.baseline.resolve(), target_is_directory=True)
    receipt = publish_baseline(native, baseline)
    full_path = root / "core-8k.json"
    if not full_path.exists():
        from_sweep(args.manifest, args.policy, baseline, full_path)
    protocol = Protocol.load(full_path)
    protocol.validate_runtime(baseline)
    _, rendered = render_questions(protocol, baseline)
    selected = []
    for benchmark in ("math500", "aime24", "aime25"):
        selected.extend([q.model_copy(update={"repeats": 1}) for q in protocol.questions
                         if q.benchmark == benchmark][:2])
    smoke = protocol.model_copy(update={"name": "SMOKE-ONLY-six-questions-8k", "questions": selected})
    write(root / "smoke-protocol.json", smoke.model_dump())
    pool = initialize(root / "pool", WorkerProfile())
    immutable_json(pool / "protocols" / f"{smoke.identity}.json", smoke.model_dump())
    task = enqueue_evaluation(Queue(pool), {"kind": "evaluate", "run_id": "SMOKE-ONLY", "step": 0, "lag": 0,
                               "training_seed": None, "model": str(baseline),
                               "checkpoint_id": receipt["checkpoint_id"], "protocol": smoke.identity}, smoke)
    report = {"prepared": True, "task": task, "protocol": protocol.identity,
              "questions": len(protocol.questions), "responses": sum(q.repeats for q in protocol.questions),
              "longest_prompt": max(map(len, rendered.values())),
              "versions": {name: version(name) for name in protocol.backend_versions}}
    write(root / "prepare-result.json", report)
    print(report, flush=True)


def stress(args):
    configure_inference_environment()
    from vllm import SamplingParams

    root = args.directory.resolve()
    protocol = Protocol.load(root / "smoke-protocol.json")
    profile = WorkerProfile()
    protocol.validate_runtime(root / "baseline")
    tok, _ = render_questions(protocol, root / "baseline")
    receipt = read(root / "baseline/evaluation-ready.json")
    with staged_model(root / "pool", root / "baseline", receipt["checkpoint_id"]) as (cached_model, _):
        engine = VLLMEngine(cached_model, protocol, profile)
        # Two full 2K prompts and two forced 8K responses exercise the worst KV case.
        unit = tok.encode(" 1 + 1 = 2.", add_special_tokens=False)
        prompt = (unit * (2048 // len(unit) + 1))[:2048]
        started = time.time()
        results = engine.llm.generate([{"prompt_token_ids": prompt}] * 2,
                                     [SamplingParams(max_tokens=8192, min_tokens=8192, ignore_eos=True,
                                                     temperature=1.0, seed=42+i) for i in range(2)], use_tqdm=False)
        if len(results) != 2 or any(len(r.outputs[0].token_ids) != 8192 for r in results):
            raise ValueError("Full-context GPU stress did not complete")
        report = {"passed": True, "prompt_tokens_each": 2048, "response_tokens_each": 8192,
                  "concurrent_sequences": 2, "seconds": time.time() - started, "hardware": engine.hardware,
                  "profile": profile.model_dump(), "benchmark_score": False}
        write(root / "stress-result.json", report)
        print(report, flush=True)


def gpu(args):
    root = args.directory.resolve()
    # Stress uses a separate process so its model allocation is released before
    # the real worker's load, grading and durable-output smoke test.
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "stress", str(root)], check=True, timeout=5400)
    work(root / "pool", "evaluate", max_tasks=9)
    tasks = Queue(root / "pool").snapshot()["tasks"]
    expected_shards = len(list(Protocol.load(root / "smoke-protocol.json").shards()))
    if len(tasks) != expected_shards or any(task["state"] != "done" for task in tasks.values()):
        raise ValueError("GPU evaluation smoke did not complete; inspect worker attempts")
    # Restarting the worker must not regenerate a completed task.
    if work(root / "pool", "evaluate", once=True)["processed_attempts"] != 0:
        raise ValueError("Completed evaluation was not idempotent")
    report = {"passed": True, "job": os.environ.get("SLURM_JOB_ID"), "restart_additional_tasks": 0,
              "stress": read(root / "stress-result.json"), "task": next(iter(tasks))}
    write(root / "gpu-result.json", report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "gpu", "stress"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--model-record", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    {"prepare": prepare, "gpu": gpu, "stress": stress}[args.phase](args)


if __name__ == "__main__":
    main()
