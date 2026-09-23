import argparse
import concurrent.futures
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import queue
import signal
import shutil
import sys
import threading
import tempfile
import time
from collections import defaultdict
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
SHARD_NODES = {0: "deep-chungus-1", 1: "deep-chungus-11"}
HARDWARE = {0: {"gpu_name_contains": "A100", "minimum_memory_gib": 37}, 1: {"gpu_name_contains": "A100", "minimum_memory_gib": 75}}
MODEL_IDS = ["qwen25-math-1.5b-base", *[f"qwen25-math-1.5b-staleness-{cap}" for cap in (2, 4, 6, 8)]]


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_package():
    path = PACKAGE / "source-manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or not {"newdispatcher.py", "gpu_devices.py"} <= manifest.get("files", {}).keys():
        raise ValueError("Partial-dispatch manifest does not cover its implementation")
    for name, expected in manifest["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or file_hash(PACKAGE / name) != expected:
            raise ValueError(f"Partial-dispatch source hash mismatch: {name}")
    return file_hash(path)


def load_source(source_root):
    root = Path(source_root).resolve()
    if "math_sweep.core" in sys.modules and not Path(sys.modules["math_sweep.core"].__file__).resolve().is_relative_to(root):
        raise ValueError("A different evaluation source is already imported")
    path = root / "evaluation/launches/15b-math-20260919/dispatch.py"
    spec = importlib.util.spec_from_file_location("frozen_math_dispatch", path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import the frozen source launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.ROOT.resolve() != root:
        raise ValueError("Frozen source root mismatch")
    return module, module.verify_package()


def partition_entries(entries, shards):
    if shards != 2:
        raise ValueError("This authorized launch has exactly two node shards")
    groups = defaultdict(list)
    for entry in entries:
        cell = entry["cell"]
        groups[(cell["profile"], cell["benchmark"])].append(entry)
    weighted = []
    seen_cells = set()
    for key, values in groups.items():
        by_model = defaultdict(set)
        for entry in values:
            cell = entry["cell"]
            if cell["cell_id"] in seen_cells:
                raise ValueError("Duplicate prepared cell in the partition")
            seen_cells.add(cell["cell_id"])
            by_model[cell["model_id"]].add(cell["budget"])
        if set(by_model) != set(MODEL_IDS) or len({tuple(sorted(budgets)) for budgets in by_model.values()}) != 1:
            raise ValueError("Each comparison group requires every model and the same token budgets")
        cost = sum(entry["cell"]["expected_responses"] * entry["cell"]["budget"] for entry in values)
        weighted.append((key, cost, values))
    loads = [0] * shards
    assigned = [[] for _ in range(shards)]
    decisions = []
    for key, cost, values in sorted(weighted, key=lambda item: (-item[1], item[0])):
        shard = min(range(shards), key=lambda index: (loads[index], index))
        loads[shard] += cost
        assigned[shard].extend(values)
        decisions.append({"profile": key[0], "benchmark": key[1], "shard": shard, "node": SHARD_NODES[shard], "maximum_output_tokens": cost, "cell_ids": sorted(entry["cell"]["cell_id"] for entry in values)})
    if not all(assigned) or sum(map(len, assigned)) != len(entries):
        raise ValueError("Partition did not assign every cell exactly once")
    return assigned, {"schema_version": 1, "shards": shards, "shard_nodes": {str(key): value for key, value in SHARD_NODES.items()}, "grouping": ["profile", "benchmark"], "groups": decisions, "maximum_output_tokens_by_shard": loads, "cell_counts": [len(items) for items in assigned]}


def derive_entry(entry, output_root, shard, package_sha256, source, usage="production"):
    original = entry["cell"]
    original_dir = Path(entry["prepared_dir"])
    original_preparation = source.read_json(original_dir / "preparation.json")
    original_preparation_sha256 = file_hash(original_dir / "preparation.json")
    if any(file_hash(original_dir / name) != sha for name, sha in original_preparation["files"].items()):
        raise ValueError("Original prepared files changed before derivation")
    cell = copy.deepcopy(original)
    hardware = dict(HARDWARE[shard])
    transition = {"kind": "normal_partial_node_hardware", "original_cell_id": original["cell_id"], "original_preparation_sha256": original_preparation_sha256, "launcher_sha256": package_sha256, "assigned_node": SHARD_NODES[shard], "shard": shard, "hardware_cohort": "A100-40GB" if shard == 0 else "A100-80GB", "usage": usage}
    cell["hardware"] = hardware
    cell["comparison_sha256"] = source.digest({"source_comparison_sha256": original["comparison_sha256"], "hardware": hardware, "assigned_node": SHARD_NODES[shard], "shard": shard})
    cell["transition"] = transition
    cell.pop("cell_id")
    cell["cell_id"] = source.digest(cell)[:24]
    allowed = {"hardware", "comparison_sha256", "transition", "cell_id"}
    if {key: value for key, value in cell.items() if key not in allowed} != {key: value for key, value in original.items() if key not in allowed}:
        raise ValueError("Derivation changed a frozen model, prompt, decoding, data or runtime field")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / cell["cell_id"]
    manifest = copy.deepcopy(original_preparation)
    manifest["derivation"] = transition
    if destination.exists():
        saved = source.read_json(destination / "preparation.json")
        if source.read_json(destination / "cell.json") != cell or saved.get("derivation") != transition:
            raise ValueError("Existing derived cell has incompatible provenance")
        if file_hash(destination / "prompts.jsonl") != original_preparation["files"]["prompts.jsonl"] or any(file_hash(destination / name) != sha for name, sha in saved["files"].items()):
            raise ValueError("Existing derived prepared files changed")
        manifest["files"] = saved["files"]
        if saved != manifest:
            raise ValueError("Existing derived manifest changed")
    else:
        staging = output_root / ".staging"
        staging.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="derive-", dir=staging) as temporary:
            directory = Path(temporary)
            source.write_json(directory / "cell.json", cell)
            shutil.copyfile(original_dir / "prompts.jsonl", directory / "prompts.jsonl")
            manifest["files"] = {name: file_hash(directory / name) for name in ("cell.json", "prompts.jsonl")}
            if manifest["files"]["prompts.jsonl"] != original_preparation["files"]["prompts.jsonl"]:
                raise ValueError("Derivation modified prompt bytes")
            source.write_json(directory / "preparation.json", manifest)
            directory.rename(destination)
    return {"source_cell_id": entry["source_cell_id"], "original_cell_id": original["cell_id"], "prepared_dir": str(destination.resolve()), "cell": cell}


def run(args, source, source_sha256, package_sha256):
    if not args.workers_from_visible or args.shards != 2 or args.shard not in SHARD_NODES:
        raise ValueError("Use --workers-from-visible with exactly one authorized shard of two")
    entries, identity = source.prepared_entries(args, source_sha256)
    shards, partition = partition_entries(entries, args.shards)
    from gpu_devices import allocation

    allocation_info = allocation(args)
    node = allocation_info["host"].split(".")[0]
    if node != SHARD_NODES[args.shard]:
        raise ValueError(f"Shard {args.shard} belongs to {SHARD_NODES[args.shard]}, not {node}")
    if allocation_info["gpu_count"] != 2 or len(set(allocation_info["worker_gpu_uuids"])) != 2:
        raise ValueError("Each authorized node shard requires exactly two distinct GPUs")
    original_results_root = args.original_results_root or args.prepared_root.parent / "results"
    shared_scope = original_results_root / ".dispatch" / identity["plan_sha256"][:24]
    scope = args.results_root / ".partial-dispatch" / identity["plan_sha256"][:24]
    shared_scope.mkdir(parents=True, exist_ok=True)
    scope.mkdir(parents=True, exist_ok=True)
    partition.update(source_identity=identity, launcher_sha256=package_sha256)
    with (shared_scope / "run.lock").open("a") as full_matrix_lock, (scope / f"shard-{args.shard}.lock").open("a") as shard_lock:
        fcntl.flock(full_matrix_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(shard_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        partition_path = scope / "partition.json"
        with (scope / "partition.lock").open("a") as partition_lock:
            fcntl.flock(partition_lock, fcntl.LOCK_EX)
            if partition_path.exists() and source.read_json(partition_path) != partition:
                raise ValueError("Saved partition belongs to different inputs, code or model-to-node assignments")
            if not partition_path.exists():
                source.write_json(partition_path, partition)
        selected = [derive_entry(entry, args.derived_prepared_root, args.shard, package_sha256, source) for entry in shards[args.shard]]
        derived_index = {"schema_version": 1, "status": "complete", "shard": args.shard, "shards": args.shards, "assigned_node": node, "hardware": HARDWARE[args.shard], "source_identity": identity, "launcher_sha256": package_sha256, "partition_sha256": file_hash(partition_path), "cells": {entry["source_cell_id"]: {"original_cell_id": entry["original_cell_id"], "cell_id": entry["cell"]["cell_id"], "preparation_sha256": file_hash(Path(entry["prepared_dir"]) / "preparation.json")} for entry in selected}}
        index_path = args.derived_prepared_root / f"preparation-index-shard-{args.shard}.json"
        if index_path.exists() and source.read_json(index_path) != derived_index:
            raise ValueError("Derived preparation index has incompatible provenance")
        source.write_json(index_path, derived_index)
        launch_id = f"{allocation_info['job_id']}-shard-{args.shard}-{time.time_ns()}"
        directory = scope / launch_id
        directory.mkdir()
        receipt_path = directory / "dispatch.json"
        receipt = {"schema_version": 1, "status": "smoke", "started_at": source.now(), "launch_id": launch_id, **identity, "partial_launcher_sha256": package_sha256, "derived_index_sha256": file_hash(index_path), "hardware_cohort": "A100-40GB" if args.shard == 0 else "A100-80GB", "partition_sha256": file_hash(partition_path), "shard": args.shard, "shards": args.shards, "expected_cells": len(selected), "allocation": allocation_info, "smoke": [], "completed": [], "priority": "normal"}
        source.write_json(receipt_path, receipt)
        children = source.Children()
        previous_handlers = {}
        receipt_lock = threading.Lock()
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, lambda number, frame: children.stop())
        try:
            candidates = {entry["cell"]["model_id"]: entry for entry in entries if entry["cell"]["profile"] == "greedy-native" and entry["cell"]["benchmark"] == "math500"}
            if set(candidates) != set(MODEL_IDS):
                raise ValueError("All five model arms require a smoke source")
            devices = allocation_info["worker_gpu_uuids"]
            smoke_jobs = [[] for _ in devices]
            for model_index, model_id in enumerate(MODEL_IDS):
                worker_index = model_index % len(devices)
                smoke_source = derive_entry(candidates[model_id], directory / "smoke-source-prepared", args.shard, package_sha256, source, usage="smoke_source")
                entry = source.create_smoke(smoke_source, directory / "smoke-prepared", allocation_info, worker_index, launch_id)
                smoke_jobs[worker_index].append(entry)

            def smoke_worker(index, device):
                for entry in smoke_jobs[index]:
                    if children.stopped.is_set():
                        raise InterruptedError("Partial dispatch stopped during model smoke checks")
                    try:
                        result = source.run_process(entry, directory / "smoke-results", device, directory / "logs" / f"gpu-{index}-smoke", children)
                        with receipt_lock:
                            receipt["smoke"].append({**result, "model_id": entry["cell"]["model_id"], "worker": index, "gpu_uuid": device})
                            source.write_json(receipt_path, receipt)
                    except BaseException:
                        children.stop()
                        raise

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
                futures = [pool.submit(smoke_worker, index, device) for index, device in enumerate(devices)]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            if children.stopped.is_set() or {item["model_id"] for item in receipt["smoke"]} != set(MODEL_IDS):
                raise InterruptedError("Not every model completed smoke validation")
            receipt.update(status="running", smoke_completed_at=source.now())
            source.write_json(receipt_path, receipt)
            pending = queue.Queue()
            for entry in sorted(selected, key=lambda item: (-item["cell"]["expected_responses"] * item["cell"]["budget"], item["cell"]["cell_id"])):
                pending.put(entry)

            def worker(index, device):
                while not children.stopped.is_set():
                    try:
                        entry = pending.get_nowait()
                    except queue.Empty:
                        return
                    try:
                        result = source.run_process(entry, args.results_root, device, directory / "logs" / f"gpu-{index}", children)
                        with receipt_lock:
                            receipt["completed"].append({**result, "worker": index, "gpu_uuid": device})
                            receipt["updated_at"] = source.now()
                            source.write_json(receipt_path, receipt)
                    except BaseException:
                        children.stop()
                        raise
                    finally:
                        pending.task_done()

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
                futures = [pool.submit(worker, index, device) for index, device in enumerate(devices)]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            if children.stopped.is_set() or len(receipt["completed"]) != len(selected):
                raise InterruptedError("Shard stopped before every production cell completed")
            receipt.update(status="complete", completed_at=source.now())
            source.write_json(receipt_path, receipt)
            return receipt
        except BaseException as error:
            children.stop()
            receipt.update(status="failed", failed_at=source.now(), error={"type": type(error).__name__, "message": str(error)})
            source.write_json(receipt_path, receipt)
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--derived-prepared-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--original-results-root", type=Path)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--workers-from-visible", action="store_true")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--gres-config", type=Path, required=True)
    parser.add_argument("--expected-gres-sha256", required=True)
    args = parser.parse_args()
    package_sha256 = verify_package()
    source, source_sha256 = load_source(args.source_root)
    result = run(args, source, source_sha256, package_sha256)
    print(json.dumps({"status": result["status"], "shard": args.shard, "cells": len(result["completed"])}), flush=True)


if __name__ == "__main__":
    main()
