import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from deepseek_study.runtime.checkpoints import atomic_write
from deepseek_study.tracking.archive import MetricMirror, sha256
from deepseek_study.config import StudyConfig
from deepseek_study.tracking.runboard import terminal_status


SUITES = {
    "bapo": ("aime2024", "aime2025"),
    "bapo_llama": ("aime2024", "aime2025", "math"),
    "m2po": ("aime2024", "aime2025", "amc2023", "amc2024", "math500", "gaokao", "minerva_math", "olympiad_bench"),
}


def import_evaluation(output, predictions, protocol, step):
    output, predictions, protocol = Path(output), Path(predictions), Path(protocol)
    if terminal_status(output) is None:
        raise ValueError("Offline evaluation import requires a terminated training run")
    marker = output / "checkpoints" / f"step_{step}" / "study/complete.json"
    if step < 1 or not marker.is_file() or json.loads(marker.read_text())["step"] != step:
        raise ValueError("Evaluation must name a completed saved checkpoint")
    settings = json.loads(protocol.read_text())
    suite = settings["suite"]
    if suite not in SUITES or set(settings["benchmarks"]) != set(SUITES[suite]):
        raise ValueError("Evaluation protocol must specify every benchmark in the selected paper suite")
    for key in ("grader_identity", "sampling", "prompt_template_sha256", "checkpoint_components_sha256"):
        if not settings.get(key):
            raise ValueError(f"Missing evaluation provenance: {key}")
    completion = json.loads(marker.read_text())
    if settings["checkpoint_components_sha256"] != completion.get("components_sha256"):
        raise ValueError("Evaluation checkpoint identity differs from the saved checkpoint")
    groups = defaultdict(dict)
    for line in predictions.read_text().splitlines():
        row = json.loads(line)
        benchmark, question, repeat = row["benchmark"], str(row["question_id"]), row["repeat"]
        if type(row["correct"]) is not bool or type(repeat) is not int or repeat < 0:
            raise ValueError("Expected boolean correctness and a zero-based integer repetition")
        key = (question, repeat)
        if key in groups[benchmark]:
            raise ValueError("Duplicate evaluation prediction")
        groups[benchmark][key] = row["correct"]
    if set(groups) != set(SUITES[suite]):
        raise ValueError("Predictions do not cover the requested benchmark suite")
    metrics = {}
    for name in SUITES[suite]:
        manifest = settings["benchmarks"][name]
        questions = [str(value) for value in manifest["question_ids"]]
        repeats = manifest["samples_per_question"]
        if not questions or len(set(questions)) != len(questions) or type(repeats) is not int or repeats < 1:
            raise ValueError("Invalid evaluation question manifest")
        if not manifest.get("dataset_revision") or not manifest.get("dataset_sha256"):
            raise ValueError("Evaluation datasets must be identified by revision and content hash")
        if suite.startswith("bapo") and repeats != 16:
            raise ValueError("The BAPO reported protocol averages 16 responses per question")
        expected = {(question, repeat) for question in questions for repeat in range(repeats)}
        if set(groups[name]) != expected:
            raise ValueError(f"Missing or unexpected evaluation predictions for {name}")
        scores = [
            sum(groups[name][(question, repeat)] for repeat in range(repeats)) / repeats for question in questions
        ]
        accuracy = sum(scores) / len(scores)
        prefix = f"{suite}/{name}"
        metrics[f"{prefix}/accuracy"] = accuracy
        metrics[f"{prefix}/accuracy_percent"] = 100 * accuracy
        metrics[f"{prefix}/questions"] = len(questions)
        metrics[f"{prefix}/responses"] = len(expected)
        if len(scores) > 1:
            variance = sum((score - accuracy) ** 2 for score in scores) / (len(scores) - 1)
            metrics[f"{prefix}/question_standard_error"] = math.sqrt(variance / len(scores))
    average = sum(metrics[f"{suite}/{name}/accuracy"] for name in SUITES[suite]) / len(SUITES[suite])
    metrics[f"{suite}/macro_accuracy"] = average
    metrics[f"{suite}/macro_accuracy_percent"] = 100 * average
    identity = hashlib.sha256((sha256(predictions) + sha256(protocol) + sha256(marker)).encode()).hexdigest()
    target = output / "evaluations" / f"{suite}-step_{step}"
    receipt = {"step": step, "suite": suite, "sha256": identity, "metrics": metrics}
    existing = target / "receipt.json"
    if existing.is_file():
        if json.loads(existing.read_text()) != receipt:
            raise FileExistsError("This checkpoint already has different evaluation results for the suite")
    else:
        atomic_write(target / "protocol.json", protocol.read_bytes())
        atomic_write(target / "predictions.jsonl", predictions.read_bytes())
        atomic_write(existing, json.dumps(receipt).encode())
    journal = output / "evaluation-metrics.jsonl"
    logged = {json.loads(line)["sha256"] for line in journal.read_text().splitlines()} if journal.exists() else set()
    if identity not in logged:
        with journal.open("a") as stream:
            stream.write(json.dumps(receipt, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    study = StudyConfig.read(output / "configs/study.json")
    MetricMirror(output, study.metrics_mirror_root).finish()
    return receipt
