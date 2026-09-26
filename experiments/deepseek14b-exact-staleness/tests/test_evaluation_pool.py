import asyncio
import ast
import gc
import json
import multiprocessing
import runpy
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from deepseek_study.evaluation.common import digest, io_slot, read, tokenizer_identity, write
from deepseek_study.evaluation.discovery import checkpoint_receipt, discover, enqueue_evaluation, register
from deepseek_study.evaluation.export import export_model, validate_export
from deepseek_study.evaluation.protocol import Protocol, Question, from_sweep, grader_identity
from deepseek_study.evaluation.grader_v3 import compare, grade
from deepseek_study.evaluation.queue import LostLease, Queue
from deepseek_study.evaluation.runner import (
    WorkerProfile, configure_inference_environment, evaluate, render_questions, summarize,
)
from deepseek_study.evaluation.service import initialize, slurm_command, work
from deepseek_study.rollouts.queue import QueueState
from deepseek_study.runtime import checkpoints


def official_checkpoint_classes():
    if sys.platform != "darwin":
        from prime_rl.trainer.ckpt import AppState, Progress

        return AppState, Progress
    # Execute the exact upstream classes without importing unrelated CUDA-only
    # optimizer kernels on macOS. Linux tests import the real module normally.
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_model_state_dict, set_state_dict
    from torch.distributed.checkpoint.stateful import Stateful

    vendor = Path(__file__).parents[1] / "vendor/prime-rl/src/prime_rl/trainer"
    source = ast.parse((vendor / "ckpt.py").read_text())
    nodes = [node for node in source.body if isinstance(node, ast.ClassDef) and node.name in {"Progress", "AppState"}]
    namespace = {**globals(), **locals(), **runpy.run_path(str(vendor / "optim/base.py")),
                 "gc": gc, "dataclass": dataclass, "asdict": asdict, "get_state_dict": get_state_dict,
                 "set_state_dict": set_state_dict, "set_model_state_dict": set_model_state_dict, "Stateful": Stateful}
    namespace["__name__"] = __name__
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                           *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(vendor / "ckpt.py"), "exec"), namespace)
    return namespace["AppState"], namespace["Progress"]


@pytest.fixture
def tiny_checkpoint(tmp_path):
    torch.manual_seed(7)
    config = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=False)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).float()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda step: 1.0)
    inputs = torch.tensor([[1, 2, 3, 4]])
    model(inputs, labels=inputs).loss.backward()
    optim.step()
    scheduler.step()
    # Actual pinned PrimeRL AppState, including optimizer/scheduler/progress.
    AppState, Progress = official_checkpoint_classes()

    checkpoint = tmp_path / "training/checkpoints/step_100"
    dcp.save({"app": AppState(model, [optim], scheduler, Progress(step=101))},
             checkpoint_id=checkpoint / "trainer", no_dist=True)
    for name in ("orchestrator/progress.pt", "rng/rank_0.pt"):
        path = checkpoint / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint-component")
    state = QueueState(0)
    state.completed_steps = 100
    state.generated_cohorts = 100
    component_hash = checkpoints.seal(checkpoint, 1)
    checkpoints.save(checkpoint / "study", state, "configuration", "identity", component_hash)
    source_model = tmp_path / "source-model"
    source_model.mkdir()
    config.save_pretrained(source_model)
    fixture = Path(__file__).parents[1] / "assets/tokenizer-fixture"
    if not (fixture / "tokenizer.json").is_file():
        pytest.skip("Pinned tokenizer fixture not downloaded")
    shutil.copyfile(fixture / "tokenizer.json", source_model / "tokenizer.json")
    tok_config = read(fixture / "tokenizer_config.json")
    tok_config["tokenizer_class"] = "TokenizersBackend"
    write(source_model / "tokenizer_config.json", tok_config)
    return checkpoint, source_model, model


@pytest.fixture
def exported(tmp_path, tiny_checkpoint):
    checkpoint, source_model, _ = tiny_checkpoint
    source = checkpoint_receipt(checkpoint)
    destination = tmp_path / "export"
    export_model(source, source_model, destination, shard_bytes=8192)
    return destination, source


def make_protocol(model, repeats=1):
    question = Question(id="math500:test/1", benchmark="math500", question="What is 1+1?", gold="2",
                        repeats=repeats, source_revision="pinned", source_sha256="a"*64, row_hash="b"*64)
    return Protocol(name="test-only", questions=[question], response_tokens=64, tokenizer=tokenizer_identity(model),
                    grader=grader_identity(), source={"test_fixture": True})


def test_official_appstate_export_exact_bf16_weights_and_logits(exported, tiny_checkpoint):
    destination, source = exported
    _, _, original = tiny_checkpoint
    loaded = AutoModelForCausalLM.from_pretrained(destination, local_files_only=True, dtype=torch.bfloat16,
                                                attn_implementation="eager")
    expected = original.to(torch.bfloat16).eval()
    assert set(loaded.state_dict()) == set(expected.state_dict())
    for key, value in expected.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key
    with torch.no_grad():
        inputs = torch.tensor([[1, 2, 3, 4]])
        assert torch.equal(expected(inputs).logits, loaded.eval()(inputs).logits)
    receipt = validate_export(destination, digest(source))
    assert receipt["model_bytes"] == sum(v.numel()*2 for v in expected.state_dict().values())
    assert all("optim" not in key for key in read(destination / "model.safetensors.index.json")["weight_map"])


def test_export_corruption_and_changed_checkpoint_rejected(exported, tiny_checkpoint, tmp_path):
    destination, source = exported
    shard = next(destination.glob("*.safetensors"))
    raw = bytearray(shard.read_bytes())
    raw[-1] ^= 1
    shard.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        validate_export(destination)
    checkpoint, source_model, _ = tiny_checkpoint
    marker = checkpoint / "study/complete.json"
    content = read(marker)
    content["lag"] = 8
    write(marker, content)
    with pytest.raises(ValueError, match="changed"):
        export_model(source, source_model, tmp_path / "changed")


def test_export_never_reads_optimizer_or_queue(tiny_checkpoint, tmp_path, monkeypatch):
    checkpoint, source_model, _ = tiny_checkpoint
    calls = []
    original_load = dcp.load

    def tracked(state_dict, **kwargs):
        assert set(state_dict) == {"app"}
        assert set(state_dict["app"]) == {"model"}
        calls.append(sum(t.numel()*t.element_size() for t in state_dict["app"]["model"].values()))
        return original_load(state_dict, **kwargs)

    monkeypatch.setattr(dcp, "load", tracked)
    (checkpoint / "study/queue.pkl").write_bytes(b"intentionally not pickle")
    export_model(checkpoint_receipt(checkpoint), source_model, tmp_path / "export", shard_bytes=8192)
    assert len(calls) > 1
    assert max(calls) <= 8192


def spec(run="a", step=100, kind="evaluate"):
    return {"kind": kind, "run_id": run, "step": step}


def test_fair_queue_oldest_per_run_and_dependencies(tmp_path):
    queue = Queue(tmp_path)
    a200 = queue.enqueue(spec("a", 200))
    a100 = queue.enqueue(spec("a", 100))
    b100 = queue.enqueue(spec("b", 100))
    assert queue.claim("evaluate", "one")["id"] == a100
    assert queue.claim("evaluate", "two")["id"] == b100
    assert queue.claim("evaluate", "three")["id"] == a200
    export = queue.enqueue(spec("c", kind="export"))
    evaluate_id = queue.enqueue(spec("c"), depends_on=export)
    assert queue.claim("evaluate", "four") is None
    claim = queue.claim("export", "cpu")
    queue.finish(claim, {"valid": True})
    assert queue.claim("evaluate", "four")["id"] == evaluate_id


def test_lease_expiry_fences_heartbeat_results_and_failure(tmp_path):
    clock = [100.0]
    queue = Queue(tmp_path, clock=lambda: clock[0])
    task_id = queue.enqueue(spec(), max_attempts=2)
    old = queue.claim("evaluate", "old", seconds=10)
    clock[0] += 11
    new = queue.claim("evaluate", "new", seconds=10)
    assert old["token"] != new["token"]
    for action in (lambda: queue.heartbeat(old), lambda: queue.finish(old, {}),
                   lambda: queue.save_sample(old, "a"*64, "raw", {}), lambda: queue.fail(old, "error")):
        with pytest.raises(LostLease):
            action()
    queue.fail(new, "persistent OOM")
    assert queue.snapshot()["tasks"][task_id]["state"] == "failed"
    assert queue.claim("evaluate", "third") is None
    queue.retry(task_id)
    assert queue.claim("evaluate", "explicit retry")


def _claim_process(root, destination):
    claim = Queue(root).claim("evaluate", str(destination))
    Path(destination).write_text(json.dumps(claim))


def test_multiple_processes_claim_exactly_once(tmp_path):
    queue = Queue(tmp_path / "queue")
    queue.enqueue(spec())
    ctx = multiprocessing.get_context("spawn")
    processes = [ctx.Process(target=_claim_process, args=(queue.root, tmp_path / str(i))) for i in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0
    assert sum(read(tmp_path / str(i)) is not None for i in range(4)) == 1


def test_idempotent_discovery_and_wrong_source_rejected(tmp_path, tiny_checkpoint, study):
    checkpoint, model, _ = tiny_checkpoint
    study = study.model_copy(update={"output_dir": checkpoint.parents[1], "prepared_model_path": model,
                                    "prompt_instruction": make_protocol(model).instruction,
                                    "checkpoint_interval": 100, "checkpoint_keep_interval": 100,
                                    "max_steps": 1000, "lag": 0})
    write(study.output_dir / "configs/study.json", study.model_dump(mode="json"))
    write(study.output_dir / "source/identity.json", {"sha256": "identity"})
    marker = read(checkpoint / "study/complete.json")
    marker["config_sha256"] = study.fingerprint()
    write(checkpoint / "study/complete.json", marker)
    protocol_path = tmp_path / "protocol.json"
    write(protocol_path, make_protocol(model).model_dump())
    root = initialize(tmp_path / "pool", WorkerProfile())
    register(root, study.output_dir, protocol_path)
    assert len(discover(root)["tasks"]) == 1
    assert len(discover(root)["tasks"]) == 1
    assert len(Queue(root).snapshot()["tasks"]) == 2
    marker["identity_sha256"] = "different-code"
    write(checkpoint / "study/complete.json", marker)
    assert "another training" in discover(root)["errors"][0]["error"]
    (checkpoint / "study/complete.json").unlink()
    assert discover(root) == {"tasks": [], "errors": []}


def test_saved_rollout_survives_failure_without_regeneration(exported, tmp_path):
    destination, source = exported
    protocol = make_protocol(destination, repeats=2)
    root = initialize(tmp_path / "pool", WorkerProfile(max_sequences=1))
    write(root / "protocols" / f"{protocol.identity}.json", protocol.model_dump())
    queue = Queue(root)
    queue.enqueue({**spec(), "model": str(destination), "protocol": protocol.identity,
                   "checkpoint_id": digest(source), "lag": 8, "training_seed": 42})
    claim = queue.claim("evaluate", "first")
    tok = AutoTokenizer.from_pretrained(destination, local_files_only=True)
    generated = tok.encode(r"</think>\boxed{2}", add_special_tokens=False)
    calls = []

    class InterruptingEngine:
        hardware = [{"synthetic": True}]

        def __init__(self, *args):
            pass

        def generate(self, requests):
            calls.append(requests)
            if len(calls) == 2:
                raise RuntimeError("injected worker interruption")
            return [{"token_ids": generated, "finish_reason": "stop"} for _ in requests]

    with pytest.raises(RuntimeError, match="interruption"):
        asyncio.run(evaluate(queue, claim, WorkerProfile(max_sequences=1), InterruptingEngine, False))
    queue.fail(claim, "interruption")
    replacement = queue.claim("evaluate", "replacement")
    result = asyncio.run(evaluate(queue, replacement, WorkerProfile(max_sequences=1), InterruptingEngine, False))
    assert len(calls) == 3  # Saved first response reused; only second response regenerated.
    assert result["metrics"]["macro_accuracy"] == 1
    assert result["metrics"]["benchmarks"]["math500"]["responses"] == 2
    assert queue.snapshot()["tasks"][claim["id"]]["state"] == "done"


def test_grader_failure_retries_saved_response_without_regeneration(exported, tmp_path, monkeypatch):
    from deepseek_study.dataset.grading import GraderFailure, GraderPool

    model, source = exported
    protocol = make_protocol(model)
    root = initialize(tmp_path / "pool", WorkerProfile())
    write(root / "protocols" / f"{protocol.identity}.json", protocol.model_dump())
    queue = Queue(root)
    task_id = queue.enqueue({**spec(), "model": str(model), "protocol": protocol.identity,
                             "checkpoint_id": digest(source), "lag": 0, "training_seed": 42})
    claim = queue.claim("evaluate", "first")
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    generated = tok.encode(r"</think>\boxed{2}", add_special_tokens=False)
    calls = []

    class Engine:
        hardware = [{"synthetic": True}]

        def __init__(self, *args):
            calls.append("load")

        def generate(self, requests):
            calls.append("generate")
            return [{"token_ids": generated, "finish_reason": "stop"} for _ in requests]

    async def failed_grader(*args):
        raise GraderFailure("injected grader timeout")

    with monkeypatch.context() as patch:
        patch.setattr(GraderPool, "call", failed_grader)
        with pytest.raises(GraderFailure, match="grader timeout"):
            asyncio.run(evaluate(queue, claim, WorkerProfile(), Engine, False))
    sample_id = next(protocol.samples())[0]
    assert queue.read_sample(task_id, sample_id, "raw")["output_token_ids"] == generated
    assert not queue.sample_path(task_id, sample_id, "grade").exists()
    assert not (root / "results" / task_id / "complete.json").exists()
    queue.fail(claim, "grader timeout")
    replacement = queue.claim("evaluate", "replacement")
    result = asyncio.run(evaluate(queue, replacement, WorkerProfile(), Engine, False))
    assert result["metrics"]["macro_accuracy"] == 1
    assert calls == ["load", "generate"]
    assert queue.snapshot()["tasks"][task_id]["state"] == "done"


def test_seed_and_protocol_identity_do_not_depend_on_run_placement(exported):
    model, _ = exported
    protocol = make_protocol(model, repeats=3)
    samples = list(protocol.samples())
    assert len({seed for *_, seed in samples}) == 3
    assert [s[3] for s in samples] == [s[3] for s in protocol.model_copy(update={"name": "renamed"}).samples()]
    assert protocol.identity != protocol.model_copy(update={"response_tokens": 8192}).identity


def test_incomplete_and_grader_error_results_cannot_publish(exported):
    model, _ = exported
    protocol = make_protocol(model)
    with pytest.raises(ValueError, match="Incomplete"):
        summarize(protocol, [])
    key = next(protocol.samples())[0]
    with pytest.raises(ValueError, match="Grader failures"):
        summarize(protocol, [{"sample_id": key, "grade": {"correct": None}}])


def test_slurm_requests_forty_gb_pool_and_preserves_shell_arguments(tmp_path):
    import subprocess

    root = initialize(tmp_path / "space ; touch SENTINEL", WorkerProfile(tensor_parallel=2))
    uv = tmp_path / "fake uv"
    uv.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@"\n')
    uv.chmod(0o755)
    name, command = slurm_command(root, "evaluate", python="/env/bin/python", uv=str(uv),
                                  account="grad-students", partition="low-priority", qos="background",
                                  excluded_nodes="deep-chungus-9,deep-chungus-11")
    assert "--gres=gpu:a100:2" in command
    assert "--exclude=deep-chungus-9,deep-chungus-11" in command
    assert not any(arg.startswith("--nodelist") for arg in command)
    assert "'" in command[-1]
    assert "--no-requeue" in command
    assert name.startswith("dse-")
    assert command[-1].startswith("set -eu\n")
    result = subprocess.run(["/bin/sh", "-c", command[-1]], cwd=tmp_path, check=True,
                            capture_output=True, text=True)
    assert result.stdout.splitlines() == ["run", "--no-project", "/env/bin/python", "-m",
                                         "deepseek_study.evaluation", "work", str(root), "--kind", "evaluate"]
    assert not (tmp_path / "SENTINEL").exists()


@pytest.mark.parametrize("gold,prediction,meta,correct", [
    ("123", "123.00000001", {"benchmark": "aime24"}, False),
    ("123", r"\frac{246}{2}", {"benchmark": "aime25"}, True),
    ("A", "a", {}, False),
    ("(1,2)", r"\{1,2\}", {}, False),
    ("(1,2)", "(2,1)", {}, False),
    (r"\{1,2,3\}", r"\{3,2,1\}", {}, True),
    (r"\{1,2,3\}", r"\{1,2\}", {}, False),
    (r"\frac{x^2-1}{x-1}", "x+1", {}, False),
    (r"\sqrt{x^2}", "x", {}, False),
    ("x+y=2", "2x+2y=4", {}, True),
    ("x=1", "x(x-1)=0", {}, False),
    (r"11\sqrt2", r"11\sqrt{2}", {}, True),
    (r"E_{1},E_{2}", r"E_1,E_2", {}, True),
    ("(3,4]", "[3,4]", {}, False),
    (r"\frac{37}{4}m", r"\frac{37m}{4}", {}, True),
])
def test_frozen_grader_regressions(gold, prediction, meta, correct):
    assert compare(gold, prediction, meta)[0] is correct


@pytest.mark.parametrize("raw,correct", [
    (r"\boxed{2}", False),
    (r"</think>2", False),
    (r"</think>\boxed{2}", True),
    (r"</think>\boxed{2} then \boxed{3}", False),
])
def test_frozen_final_answer_extraction(raw, correct):
    assert grade({"raw": raw, "gold": "2"})["correct"] is correct


def test_freeze_sweep_applies_only_hash_matched_corrections_and_exclusions(tmp_path, tiny_checkpoint):
    _, model, _ = tiny_checkpoint
    rows = [{"id": f"math500:{i}", "benchmark": "math500", "question": "Question", "gold": "2", "attempts": 4,
             "meta": {}, "source_revision": "pin", "source_sha256": "a"*64, "row_hash": digest(i)} for i in range(3)]
    manifest = {"problems": rows}
    manifest["experiment_hash"] = digest(manifest)
    policy = {"experiment_hash": manifest["experiment_hash"],
              "corrections": {"math500:0": {"row_hash": rows[0]["row_hash"], "original_gold": "2", "gold": "3"}},
              "exclusions": {"math500:1": {"row_hash": rows[1]["row_hash"], "reason": "invalid question"}},
              "format_hints": {"math500:2": {"row_hash": rows[2]["row_hash"], "answer_labels": ["x"]}}}
    manifest_path, policy_path = tmp_path / "manifest.json", tmp_path / "policy.json"
    write(manifest_path, manifest)
    write(policy_path, policy)
    protocol = from_sweep(manifest_path, policy_path, model, tmp_path / "protocol.json", ["math500"])
    assert [q.id for q in protocol.questions] == ["math500:0", "math500:2"]
    assert protocol.questions[0].gold == "3"
    assert protocol.questions[1].meta == {"answer_labels": ["x"]}
    policy["corrections"]["math500:0"]["row_hash"] = "bad"
    write(policy_path, policy)
    with pytest.raises(ValueError, match="source mismatch"):
        from_sweep(manifest_path, policy_path, model, tmp_path / "wrong.json", ["math500"])


def _save_sharded(rank, rendezvous, weights, output):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,))
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model = {k: distribute_tensor(v, mesh, [Shard(0)]) for k, v in state.items()}
        dcp.save({"app": {"model": model, "optimizers": {"sentinel": torch.zeros(17)}}}, checkpoint_id=output)
    finally:
        dist.destroy_process_group()


def test_two_rank_sharded_checkpoint_consolidation(tiny_checkpoint, tmp_path):
    checkpoint, model_source, model = tiny_checkpoint
    weights = tmp_path / "expected.pt"
    torch.save(model.state_dict(), weights)
    shutil.rmtree(checkpoint / "trainer")
    ctx = multiprocessing.get_context("spawn")
    processes = [ctx.Process(target=_save_sharded, args=(rank, tmp_path / "rendezvous", weights,
                                                        checkpoint / "trainer")) for rank in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=90)
        assert process.exitcode == 0
    marker = read(checkpoint / "study/complete.json")
    marker["components_sha256"] = checkpoints.seal(checkpoint, 1)
    write(checkpoint / "study/complete.json", marker)
    destination = tmp_path / "resharded-export"
    export_model(checkpoint_receipt(checkpoint), model_source, destination, shard_bytes=8192)
    loaded = AutoModelForCausalLM.from_pretrained(destination, local_files_only=True, dtype=torch.bfloat16)
    for key, value in model.state_dict().items():
        assert torch.equal(value.to(torch.bfloat16), loaded.state_dict()[key]), key


def test_supervisor_kills_stalled_attempt_and_leaves_retryable_task(tmp_path):
    root = initialize(tmp_path / "pool", WorkerProfile())
    queue = Queue(root)
    task_id = queue.enqueue(spec())
    assert work(root, "evaluate", once=True, stall_seconds=0) == {"processed_attempts": 1}
    task = queue.snapshot()["tasks"][task_id]
    assert task["state"] == "queued"
    assert task["attempts"] == 1
    assert "No durable progress" in task["errors"][0]["error"]


def test_collection_uses_checkpoint_step_and_excludes_pending_results(tmp_path):
    from deepseek_study.evaluation.report import collect

    queue = Queue(tmp_path)
    task_spec = {**spec(step=300), "lag": 256, "training_seed": 42, "protocol": "p", "checkpoint_id": "c"}
    queue.enqueue(task_spec)
    claim = queue.claim("evaluate", "one")
    queue.finish(claim, {"spec": task_spec, "protocol": {"response_tokens": 8192},
                         "metrics": {"benchmarks": {"math500": {"accuracy": 0.7}}}})
    queue.enqueue({**task_spec, "step": 400})
    result = collect([tmp_path])
    assert len(result["rows"]) == 1
    assert result["rows"][0]["training_step"] == 300
    assert result["rows"][0]["stale_updates"] == 44
    assert result["pending"][0]["step"] == 400


def test_collection_memory_does_not_grow_with_duplicate_protocols(tmp_path):
    import tracemalloc
    from deepseek_study.evaluation.report import collect

    # A frozen full benchmark is roughly 400KB and is repeated in every shard.
    # Simulate many completed checkpoints without running generation again.
    tasks = {}
    protocol = {"response_tokens": 8192, "question_fixture": "x" * 400_000}
    for index in range(60):
        task_spec = {**spec(run=f"run-{index}"), "lag": 0, "training_seed": 42,
                     "protocol": "same-protocol", "checkpoint_id": f"checkpoint-{index}"}
        task_id = digest(task_spec)
        receipt = {"spec": task_spec, "protocol": protocol,
                   "metrics": {"benchmarks": {"math500": {"accuracy": 0.5}}}}
        write(tmp_path / "results" / task_id / "complete.json", receipt)
        tasks[task_id] = {"spec": task_spec, "state": "done", "depends_on": None,
                          "finished": 1, "receipt_sha256": digest(receipt)}
    write(tmp_path / "queue.json", {"format": 1, "tasks": tasks, "served": {}, "counter": 0})
    tracemalloc.start()
    try:
        result = collect([tmp_path])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert len(result["rows"]) == 60
    assert peak < 8 * 1024**2, f"Collector retained duplicate protocols: peak={peak} bytes"


def test_sample_corruption_cannot_silently_change_grades(tmp_path):
    queue = Queue(tmp_path)
    task_id = queue.enqueue(spec())
    claim = queue.claim("evaluate", "worker")
    sample_id = "a"*64
    queue.save_sample(claim, sample_id, "raw", {"raw": "original"})
    path = queue.sample_path(task_id, sample_id, "raw")
    record = read(path)
    record["data"]["raw"] = "modified"
    write(path, record)
    with pytest.raises(ValueError, match="checksum"):
        queue.read_sample(task_id, sample_id, "raw")


def test_prompt_tokens_match_training_renderer_exactly(exported):
    from renderers import create_renderer
    from renderers.configs import DefaultRendererConfig

    model, _ = exported
    protocol = make_protocol(model)
    tok, rendered = render_questions(protocol, model)
    renderer = create_renderer(tok, DefaultRendererConfig())
    for question in protocol.questions:
        prompt = question.question + "\n\n" + protocol.instruction
        assert rendered[question.id] == renderer.render_ids([{"role": "user", "content": prompt}],
                                                             add_generation_prompt=True)


def test_changed_source_or_tokenizer_requires_new_protocol(exported):
    model, _ = exported
    protocol = make_protocol(model)
    changed = protocol.model_copy(update={"implementation": {"runner.py": "edited"}})
    with pytest.raises(ValueError, match="Evaluator source changed"):
        changed.validate_runtime(model, backend=False)
    changed = protocol.model_copy(update={"tokenizer": {"tokenizer.json": "edited"}})
    with pytest.raises(ValueError, match="tokenizer/config"):
        changed.validate_runtime(model, backend=False)


def test_io_slot_does_not_swallow_operation_errors(tmp_path):
    with pytest.raises(BlockingIOError, match="operation failed"):
        with io_slot(tmp_path):
            raise BlockingIOError("operation failed")


def test_transient_scheduler_failure_is_reported_without_losing_queue(tmp_path, monkeypatch):
    import subprocess
    from deepseek_study.evaluation import service

    root = initialize(tmp_path / "pool", WorkerProfile())
    task = Queue(root).enqueue(spec())

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["squeue"])

    monkeypatch.setattr(service, "reconcile_slurm", fail)
    status = service.serve(root, once=True, slurm={"limits": {"export": 1, "evaluate": 2}})
    assert status["scheduler_error"]["type"] == "CalledProcessError"
    assert Queue(root).snapshot()["tasks"][task]["state"] == "queued"


def test_verified_local_cache_reuses_bytes_and_does_not_evict_active_model(exported, tmp_path):
    from deepseek_study.evaluation.cache import prune, staged_model

    model, source = exported
    directory = tmp_path / "local-cache"
    with staged_model(tmp_path / "pool", model, digest(source), directory, False) as (cached, receipt):
        assert cached != model
        assert validate_export(cached) == receipt
        prune(directory, keep=0)
        assert cached.exists()
    moved = model.with_name("moved-export")
    model.rename(moved)
    with staged_model(tmp_path / "pool", model, digest(source), directory, False) as (reused, _):
        assert reused == cached
    prune(directory, keep=0)
    assert not cached.exists()


def test_cache_does_not_publish_corrupt_model(exported, tmp_path):
    from deepseek_study.evaluation.cache import staged_model

    model, source = exported
    shard = next(model.glob("*.safetensors"))
    original = shard.read_bytes()
    shard.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="checksum"):
        with staged_model(tmp_path / "pool", model, digest(source), cache, False):
            pytest.fail("Corrupt cache published")
    assert not (cache / digest(source)).exists()
    assert not list(cache.glob(".partial-*"))


def test_shards_cover_every_repeat_once_and_only_publish_complete_checkpoint(exported, tmp_path):
    from deepseek_study.evaluation.report import collect

    model, source = exported
    base = make_protocol(model, repeats=2)
    questions = [base.questions[0].model_copy(update={"id": f"math500:{i}", "gold": str(2 if i < 2 else 3)})
                 for i in range(3)]
    protocol = base.model_copy(update={"questions": questions, "questions_per_task": 1})
    root = initialize(tmp_path / "pool", WorkerProfile())
    write(root / "protocols" / f"{protocol.identity}.json", protocol.model_dump())
    queue = Queue(root)
    task_spec = {**spec(), "model": str(model), "protocol": protocol.identity, "checkpoint_id": digest(source),
                 "lag": 8, "training_seed": 42}
    task_ids = enqueue_evaluation(queue, task_spec, protocol)
    assert len(task_ids) == 3
    assert enqueue_evaluation(queue, task_spec, protocol) == task_ids
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    generated = tok.encode(r"</think>\boxed{2}", add_special_tokens=False)
    seeds = []

    class Engine:
        hardware = [{"synthetic": True}]

        def __init__(self, *args):
            pass

        def generate(self, requests):
            seeds.extend(request["seed"] for request in requests)
            return [{"token_ids": generated, "finish_reason": "stop"} for _ in requests]

    for index in range(3):
        claim = queue.claim("evaluate", f"worker-{index}")
        asyncio.run(evaluate(queue, claim, WorkerProfile(), Engine, False))
        if index < 2:
            assert collect([root])["rows"] == []
    result = collect([root])
    assert not result["pending"]
    assert len(result["rows"]) == 1
    assert result["rows"][0]["accuracy"] == pytest.approx(2/3)
    assert result["rows"][0]["responses"] == 6
    assert result["rows"][0]["questions"] == 3
    assert sorted(seeds) == sorted(seed for *_, seed in protocol.samples())


def test_inference_setup_uses_pinned_prime_runner_and_fp32_output(monkeypatch):
    import os
    from prime_rl.configs.inference import InferenceConfig

    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    additional = configure_inference_environment()
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert additional == InferenceConfig().to_namespace().additional_config
    assert additional["fp32_lm_head"] is True
