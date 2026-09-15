import argparse
import json
import math
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from opik import Opik

stopping = False
TRAINER_PREFIXES = (
    "approx_kl/",
    "clip_fraction/",
    "entropy/",
    "importance_ratio/",
    "kl_ent_ratio/",
    "loss/",
    "mismatch_kl/",
    "optim/grad_norm",
    "optim/lr",
    "perf/mfu",
    "perf/peak_memory",
    "perf/throughput",
    "time/broadcast_weights",
)
INFERENCE_TOKEN_SIGNALS = ("generation_tokens_total", "prompt_tokens_total")
INFERENCE_LATENCY_SIGNALS = (
    "e2e_request_latency_seconds",
    "inter_token_latency_seconds",
    "time_to_first_token_seconds",
)
INFERENCE_QUEUE_SIGNALS = ("num_requests_running", "num_requests_waiting")
QUALITY_MARKERS = (
    "/avg@",
    "/dispatch_failure/mean",
    "/has_error/mean",
    "/is_truncated/mean",
    "/num_output_tokens/",
    "/pass@",
    "/pass^",
    "/reward/",
    "/rewards/",
    "/solved_all",
    "/solved_none",
    "/solved_some",
)


def stop(*_) -> None:
    global stopping
    stopping = True


def selected_metric(key: str) -> bool:
    if key.startswith(TRAINER_PREFIXES):
        return True
    if key.startswith("train/agg/"):
        return any(marker in key for marker in QUALITY_MARKERS)
    if key.startswith("eval/"):
        return any(marker in key for marker in QUALITY_MARKERS) or key.endswith(
            "/policy_version"
        )
    if key.startswith(("off_policy/", "progress/")):
        return True
    if not key.startswith("inference/agg/"):
        return False
    if any(signal in key for signal in INFERENCE_TOKEN_SIGNALS):
        return ":rate/" in key and key.endswith("/sum")
    if any(signal in key for signal in INFERENCE_LATENCY_SIGNALS):
        return ":mean/" in key and key.endswith("/mean")
    if any(signal in key for signal in INFERENCE_QUEUE_SIGNALS):
        return key.endswith("/sum")
    return False


def scalar_metrics(row: dict) -> dict[str, float]:
    metrics = {}
    for key, value in row.items():
        if not selected_metric(key) or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            metrics[key] = float(value)
    derived = {}
    for key, value in metrics.items():
        trainer_aliases = {
            "entropy/all/mean": "train/entropy",
            "loss/mean": "train/loss",
            "optim/grad_norm": "train/grad_norm",
            "optim/lr": "train/learning_rate",
            "perf/mfu": "train/mfu",
            "perf/throughput": "train/throughput",
        }
        if key in trainer_aliases:
            derived[trainer_aliases[key]] = value
        if key.startswith("importance_ratio/"):
            derived[f"train/{key}"] = value
        if key.startswith("clip_fraction/"):
            derived[f"train/{key}"] = value
        if key.startswith("train/agg/effective/") and key.endswith(
            "/reward/mean"
        ):
            derived["train/reward"] = value
            derived["train/rewards/accuracy"] = value
            derived["rollouts/correct_fraction"] = value
            if 0.0 <= value <= 1.0:
                derived["train/reward_std"] = math.sqrt(value * (1.0 - value))
        if key.startswith("train/agg/effective/") and "/num_output_tokens/" in key:
            statistic = key.rsplit("/", 1)[-1]
            derived[f"train/completion_length/{statistic}"] = value
        if key.startswith("train/agg/all/"):
            aliases = {
                "solved_none": "rollouts/all_zero_group_fraction",
                "solved_all": "rollouts/all_one_group_fraction",
                "solved_some": "rollouts/mixed_group_fraction",
            }
            for suffix, name in aliases.items():
                if key.endswith(f"/{suffix}"):
                    derived[name] = value
        if key.startswith("eval/") and "/effective/" in key:
            parts = key.split("/")
            source = parts[1]
            statistic = parts[-1]
            benchmark = source.removesuffix("-pass1").removesuffix("-sampled")
            if statistic.startswith(("avg@", "pass@", "pass^")):
                if source.endswith("-sampled") and statistic not in {
                    "avg@8",
                    "pass@8",
                }:
                    benchmark = f"{benchmark}/sampled"
                name = "mean_accuracy" if statistic == "avg@8" else statistic
                derived[f"eval/{benchmark}/{name}"] = value
    metrics.update(derived)
    return metrics


def save_state(path: Path, offset: int, last_inference_time: float) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"last_inference_time": last_inference_time, "offset": offset},
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(path)


def flush(client: Opik) -> None:
    if not client.flush(timeout=30):
        raise RuntimeError("Opik flush did not complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--prime-commit", required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    client = Opik(project_name=args.project, workspace=args.workspace)
    now = datetime.now(timezone.utc)
    client.trace(
        name="prime-rl-sidecar-start",
        start_time=now,
        end_time=now,
        input={"run": args.run_name},
        output={"status": "ready"},
        metadata={"attempt": args.attempt_id, "prime_commit": args.prime_commit},
        tags=["prime-rl", "grpo", "qwen3-14b", args.attempt_id],
        feedback_scores=[{"name": "observer/preflight", "value": 1.0}],
    )
    flush(client)
    args.ready_file.write_text("ready\n")
    offset = 0
    last_inference_time = 0.0
    if args.state.exists():
        state = json.loads(args.state.read_text())
        offset = int(state.get("offset", 0))
        last_inference_time = float(state.get("last_inference_time", 0.0))
    while True:
        pending_offset = None
        partial_tail = False
        traced = False
        if args.metrics.exists():
            size = args.metrics.stat().st_size
            if offset > size:
                offset = 0
            with args.metrics.open() as source:
                source.seek(offset)
                while True:
                    line = source.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):
                        partial_tail = True
                        break
                    next_offset = source.tell()
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise RuntimeError(
                            f"Malformed metrics JSONL record ending at byte {next_offset}"
                        ) from error
                    metrics = scalar_metrics(row)
                    timestamp = datetime.fromtimestamp(
                        float(row.get("time", time.time())), timezone.utc
                    )
                    is_inference = row.get("producer") == "orch" and row.get("step") is None
                    if is_inference and timestamp.timestamp() - last_inference_time < 30:
                        metrics = {}
                    if metrics:
                        client.trace(
                            name="prime-rl-metrics",
                            start_time=timestamp,
                            end_time=timestamp,
                            input={"step": row.get("step")},
                            output=metrics,
                            metadata={
                                "run": args.run_name,
                                "attempt": args.attempt_id,
                                "producer": row.get("producer"),
                                "prime_commit": args.prime_commit,
                            },
                            tags=[
                                "prime-rl",
                                "grpo",
                                "qwen3-14b",
                                args.attempt_id,
                            ],
                            feedback_scores=[
                                {"name": key, "value": value}
                                for key, value in metrics.items()
                            ],
                        )
                        traced = True
                        if is_inference:
                            last_inference_time = timestamp.timestamp()
                    pending_offset = next_offset
        if pending_offset is not None:
            if traced:
                flush(client)
            offset = pending_offset
            save_state(args.state, offset, last_inference_time)
        if stopping or args.stop_file.exists():
            if partial_tail:
                raise RuntimeError("Metrics JSONL ended with an incomplete record")
            break
        time.sleep(2)
    flush(client)


if __name__ == "__main__":
    main()
