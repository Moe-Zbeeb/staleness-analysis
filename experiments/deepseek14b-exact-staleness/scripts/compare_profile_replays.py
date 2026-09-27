import argparse
import copy
import json
import math
import statistics
from pathlib import Path

import numpy as np


def compare_configs(baseline, candidate, allowed_model_changes=()):
    allowed = set(allowed_model_changes)
    if allowed - {"ac", "reshard_after_forward", "compile"}:
        raise ValueError("Only documented model performance settings may differ")
    left, right = copy.deepcopy(baseline), copy.deepcopy(candidate)
    left.pop("output_dir")
    right.pop("output_dir")
    changes = {}
    for key in allowed:
        values = [left["model"].pop(key, None), right["model"].pop(key, None)]
        if values[0] != values[1]:
            changes[key] = values
    if left != right:
        raise ValueError("Trainer configurations differ beyond explicitly allowed performance settings")
    return changes


def compare(baseline, candidate, ranks, steps, allowed_model_changes=()):
    static_columns = ("token_id", "position", "behavior_logp", "advantage", "response_lengths")
    float_columns = ("current_logp", "entropy")
    mask_columns = ("surrogate_clipped", "zero_policy_signal")
    records = []
    environment_changes = []
    model_changes = []
    for rank in range(ranks):
        receipts = [json.loads((root / f"rank-{rank}" / "timings.json").read_text()) for root in (baseline, candidate)]
        if receipts[0]["grid_sha256"] != receipts[1]["grid_sha256"]:
            raise ValueError("Replay grids differ")
        if any(receipt["steps"] != steps or receipt["rank"] != rank for receipt in receipts):
            raise ValueError("Replay receipt does not match the requested comparison")
        if receipts[0]["micro_batches"] != receipts[1]["micro_batches"]:
            raise ValueError("Replay micro-batch counts differ")
        for receipt in receipts:
            finished = [row["step"] for row in receipt["rows"] if row["phase"] == "optimizer"]
            if finished != list(range(1, steps + 1)):
                raise ValueError("Replay did not complete every requested optimizer update")
        environments = [receipt["environment"] for receipt in receipts]
        changes = {
            key: [environments[0].get(key), environments[1].get(key)]
            for key in sorted(environments[0].keys() | environments[1].keys())
            if environments[0].get(key) != environments[1].get(key)
        }
        environment_changes.append({"rank": rank, "changes": changes})
        configs = [json.loads((root / f"rank-{rank}" / "config.json").read_text()) for root in (baseline, candidate)]
        model_changes.append({"rank": rank, "changes": compare_configs(configs[0], configs[1], allowed_model_changes)})
        for step in range(1, steps + 1):
            relative = Path("trainer/paper/tokens") / f"step_{step}" / f"rank_{rank}.npz"
            with (
                np.load(baseline / relative, allow_pickle=False) as left,
                np.load(candidate / relative, allow_pickle=False) as right,
            ):
                if set(left.files) != set(right.files):
                    raise ValueError("Token archive schemas differ")
                if any(
                    left[key].shape != right[key].shape or left[key].dtype != right[key].dtype for key in left.files
                ):
                    raise ValueError("Token archive shapes or dtypes differ")
                record = {"rank": rank, "step": step, "tokens": len(left["token_id"])}
                record["static_columns_equal"] = all(np.array_equal(left[key], right[key]) for key in static_columns)
                record["numerical"] = {}
                for key in float_columns:
                    a, b = left[key], right[key]
                    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                        raise ValueError("Model outputs are misaligned or non-finite")
                    error = np.abs(a.astype(np.float64) - b.astype(np.float64))
                    record["numerical"][key] = {
                        "bitwise_equal": bool(np.array_equal(a, b)),
                        "different_tokens": int(np.count_nonzero(a != b)),
                        "max_abs_error": float(error.max(initial=0)),
                        "mean_abs_error": float(error.mean()) if len(error) else 0,
                        "p99_abs_error": float(np.quantile(error, 0.99)) if len(error) else 0,
                    }
                record["mask_differences"] = {
                    key: int(np.count_nonzero(left[key] != right[key])) for key in mask_columns
                }
                records.append(record)
    metric_sets = []
    for root in (baseline, candidate):
        merged = {step: {} for step in range(1, steps + 1)}
        for line in (root / "trainer/metrics.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row.get("step") in merged:
                merged[row["step"]].update(row)
        metric_sets.append(merged)
    timing_rows = []
    for step in range(1, steps + 1):
        left, right = [metrics[step] for metrics in metric_sets]
        for values in (left, right):
            for key in ("time/forward_backward", "optim/grad_norm", "loss/mean", "perf/peak_memory"):
                if key not in values or not math.isfinite(values[key]):
                    raise ValueError(f"Replay metric is missing or non-finite: step {step}, {key}")
            if values["time/forward_backward"] <= 0:
                raise ValueError("Replay timing must be positive")
        timing_rows.append(
            {
                "step": step,
                "baseline_seconds": left["time/forward_backward"],
                "candidate_seconds": right["time/forward_backward"],
                "speedup": left["time/forward_backward"] / right["time/forward_backward"],
                "baseline_grad_norm": left.get("optim/grad_norm"),
                "candidate_grad_norm": right.get("optim/grad_norm"),
                "baseline_loss": left.get("loss/mean"),
                "candidate_loss": right.get("loss/mean"),
                "baseline_peak_memory": left["perf/peak_memory"],
                "candidate_peak_memory": right["perf/peak_memory"],
            }
        )
    return {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "ranks": ranks,
        "steps": steps,
        "static_columns_equal": all(row["static_columns_equal"] for row in records),
        "outputs_bitwise_equal": all(value["bitwise_equal"] for row in records for value in row["numerical"].values()),
        "mask_differences": {key: sum(row["mask_differences"][key] for row in records) for key in mask_columns},
        "environment_changes": environment_changes,
        "model_changes": model_changes,
        "median_step_speedup": statistics.median(row["speedup"] for row in timing_rows),
        "timings": timing_rows,
        "token_comparisons": records,
        "production_speedup_established": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-model-change", action="append", choices=("ac", "reshard_after_forward", "compile"), default=[]
    )
    args = parser.parse_args()
    if args.ranks < 2 or not 1 <= args.steps <= 10:
        raise ValueError("Expected a bounded distributed replay")
    result = compare(args.baseline, args.candidate, args.ranks, args.steps, args.allow_model_change)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("static_columns_equal", "outputs_bitwise_equal", "mask_differences", "median_step_speedup")
            }
        )
    )
    if not result["static_columns_equal"]:
        raise RuntimeError("Replay changed archived training inputs")


if __name__ == "__main__":
    main()
