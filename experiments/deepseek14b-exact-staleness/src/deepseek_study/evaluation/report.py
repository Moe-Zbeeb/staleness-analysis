from collections import defaultdict

from .common import digest, read
from .protocol import Protocol
from .queue import Queue
from .runner import metrics_from_questions


def collect(roots):
    rows, pending, groups = [], [], defaultdict(dict)
    protocols = {}
    for root in roots:
        queue = Queue(root)
        for task_id, task in queue.snapshot()["tasks"].items():
            spec = task["spec"]
            if spec["kind"] != "evaluate":
                continue
            group = spec.get("group", task_id)
            if task["state"] != "done":
                pending.append({"task": task_id, "group": group, "run_id": spec["run_id"], "step": spec["step"],
                                "state": task["state"], "blocked_by": task.get("blocked_by"), "errors": task["errors"]})
                continue
            receipt = read(queue.root / "results" / task_id / "complete.json")
            if digest(receipt) != task["receipt_sha256"] or receipt["spec"] != spec:
                raise ValueError("Completed evaluation receipt changed")
            # Every shard carries the full frozen question set for auditability.
            # Share it in memory: dozens of runs must not retain thousands of
            # duplicate protocol dictionaries in the lightweight coordinator.
            protocol = protocols.setdefault(spec["protocol"], receipt["protocol"])
            if protocol != receipt["protocol"]:
                raise ValueError("Conflicting contents for the same evaluation protocol")
            receipt["protocol"] = protocol
            previous = groups[group].get(task_id)
            if previous and previous[0] != receipt:
                raise ValueError("Conflicting copies of a completed evaluation")
            groups[group][task_id] = (receipt, task["finished"])
    for group, tasks in groups.items():
        receipts = list(tasks.values())
        first, _ = receipts[0]
        spec = first["spec"]
        expected = spec.get("shards_total", 1)
        if len(receipts) != expected:
            continue
        if "shard" in spec:
            if {r["spec"]["shard"] for r, _ in receipts} != set(range(expected)):
                raise ValueError("Duplicate or missing checkpoint shard")
            protocol = Protocol.model_validate(first["protocol"])
            questions = [q for receipt, _ in receipts for q in receipt["metrics"]["question_metrics"]]
            expected_questions = {q.id: q for q in protocol.questions}
            if len(questions) != len(expected_questions) or {q["question_id"] for q in questions} != set(expected_questions):
                raise ValueError("Incomplete or overlapping benchmark question coverage")
            for receipt, _ in receipts:
                if receipt["protocol"] != first["protocol"] or receipt["spec"]["checkpoint_id"] != spec["checkpoint_id"]:
                    raise ValueError("Mixed checkpoint or protocol in evaluation shards")
            for question in questions:
                reference = expected_questions[question["question_id"]]
                if question["responses"] != reference.repeats or question["benchmark"] != reference.benchmark:
                    raise ValueError("Incomplete repetitions or changed benchmark")
            metrics = metrics_from_questions(questions)
        else:
            metrics = first["metrics"]
        for benchmark, values in metrics["benchmarks"].items():
            rows.append({
                "run_id": spec["run_id"], "training_step": spec["step"], "lag": spec["lag"],
                "training_seed": spec["training_seed"], "benchmark": benchmark, "protocol": spec["protocol"],
                "checkpoint_id": spec["checkpoint_id"], "group": group,
                "response_token_cap": first["protocol"]["response_tokens"],
                "stale_updates": max(0, spec["step"] - spec["lag"]) if spec["lag"] else 0,
                "evaluation_completed_at": max(finished for _, finished in receipts), **values,
            })
    return {"rows": sorted(rows, key=lambda r: (r["run_id"], r["training_step"], r["benchmark"])),
            "pending": pending, "metric": "mean per-question sampled pass@1"}
