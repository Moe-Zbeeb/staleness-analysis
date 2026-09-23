import argparse
import concurrent.futures
import copy
import csv
import fcntl
import io
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
EVALUATION = PACKAGE.parents[1]
ROOT = EVALUATION.parent
sys.path.insert(0, str(EVALUATION))

from math_sweep.core import digest, file_hash, load_catalogs, make_plan, read_json, read_jsonl, validate_records, write_json


MODEL_IDS = ["qwen25-math-1.5b-base", *[f"qwen25-math-1.5b-staleness-{cap}" for cap in (2, 4, 6, 8)]]
PROFILES = ["greedy-native", "sampled-native"]


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_package():
    path = PACKAGE / "source-manifest.json"
    manifest = read_json(path)
    required = {str((PACKAGE / "dispatch.py").relative_to(ROOT)), "evaluation/sweep.py", "evaluation/models.json", "evaluation/profiles.json", "evaluation/benchmarks.json"}
    required.update(str(item.relative_to(ROOT)) for item in (EVALUATION / "math_sweep").glob("*.py"))
    if manifest.get("schema_version") != 1 or not required <= manifest.get("files", {}).keys():
        raise ValueError("Launch package manifest does not cover the evaluation implementation and catalogs")
    for name, sha in manifest["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or file_hash(ROOT / name) != sha:
            raise ValueError(f"Launch source hash mismatch: {name}")
    return file_hash(path)


def load_plan(path):
    plan = read_json(path)
    expected = make_plan(*load_catalogs(EVALUATION), selected_models=MODEL_IDS, selected_profiles=PROFILES)
    if plan != expected:
        raise ValueError("Launch requires exactly the frozen default native plan for 1.5B base and staleness 2/4/6/8")
    if any(cell["availability"] != "pinned" for cell in plan["cells"]):
        raise ValueError("The requested matrix contains an unpinned model")
    return plan


def binding(args, package_sha256):
    data_receipt = args.data.with_suffix(".receipt.json")
    if read_json(data_receipt)["sha256"] != file_hash(args.data):
        raise ValueError("Prepared data receipt mismatch")
    return {"plan_sha256": file_hash(args.plan), "data_sha256": file_hash(args.data), "data_receipt_sha256": file_hash(data_receipt), "package_sha256": package_sha256}


def validate_entry(entry, prepared_root, source_id):
    directory = prepared_root / entry["cell_id"]
    preparation = read_json(directory / "preparation.json")
    cell = read_json(directory / "cell.json")
    if entry["source_cell_id"] != source_id or preparation["source_cell_id"] != source_id or cell["cell_id"] != entry["cell_id"]:
        raise ValueError("Preparation index identity mismatch")
    if entry["preparation_sha256"] != file_hash(directory / "preparation.json"):
        raise ValueError("Indexed preparation has changed")
    if any(file_hash(directory / name) != sha for name, sha in preparation["files"].items()):
        raise ValueError("Indexed prepared input has changed")
    if cell["cell_id"] != digest({key: value for key, value in cell.items() if key != "cell_id"})[:24]:
        raise ValueError("Indexed cell digest mismatch")
    return directory


def prepare(args, package_sha256):
    from math_sweep.prepare import prepare_cell

    plan = load_plan(args.plan)
    identity = binding(args, package_sha256)
    args.prepared_root.mkdir(parents=True, exist_ok=True)
    index_path = args.prepared_root / "preparation-index.json"
    with (args.prepared_root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index = read_json(index_path) if index_path.exists() else {"schema_version": 1, **identity, "status": "preparing", "cells": {}}
        if any(index.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing preparation index belongs to different frozen inputs")
        for cell in plan["cells"]:
            source_id = cell["cell_id"]
            if source_id in index["cells"]:
                validate_entry(index["cells"][source_id], args.prepared_root, source_id)
                continue
            directory = prepare_cell(cell, args.data, args.prepared_root, args.tokenizer_root)
            index["cells"][source_id] = {"source_cell_id": source_id, "cell_id": directory.name, "preparation_sha256": file_hash(directory / "preparation.json")}
            index["updated_at"] = now()
            write_json(index_path, index)
            print(json.dumps({"phase": "prepare", "cells": len(index["cells"]), "total": len(plan["cells"]), "cell_id": directory.name}), flush=True)
        index.update(status="complete", completed_at=now())
        write_json(index_path, index)
    return index


def prepared_entries(args, package_sha256):
    plan = load_plan(args.plan)
    identity = binding(args, package_sha256)
    index_path = args.prepared_root / "preparation-index.json"
    index = read_json(index_path)
    if index.get("status") != "complete" or any(index.get(key) != value for key, value in identity.items()):
        raise ValueError("Complete preparation for this exact launch is required before GPU dispatch")
    if set(index["cells"]) != {cell["cell_id"] for cell in plan["cells"]}:
        raise ValueError("Preparation index does not contain exactly the full requested matrix")
    entries = []
    for cell in plan["cells"]:
        directory = validate_entry(index["cells"][cell["cell_id"]], args.prepared_root, cell["cell_id"])
        entries.append({"source_cell_id": cell["cell_id"], "prepared_dir": str(directory.resolve()), "cell": read_json(directory / "cell.json")})
    return entries, {**identity, "preparation_index_sha256": file_hash(index_path)}


def canonical_devices(visible, inventory, allocated_count):
    devices = visible.split(",")
    if not devices or any(not device or device.strip() != device for device in devices) or len(set(devices)) != len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must expose distinct devices")
    if allocated_count != len(inventory) or len(devices) != allocated_count:
        raise ValueError("Full-node allocation, physical inventory and GPU visibility disagree")
    physical = {str(item["index"]): item["uuid"] for item in inventory}
    if len(physical) != allocated_count or len(set(physical.values())) != allocated_count or not all(value.startswith("GPU-") for value in physical.values()):
        raise ValueError("Invalid or duplicate physical GPU UUID inventory")
    if all(device.isdigit() for device in devices):
        if set(devices) != physical.keys():
            raise ValueError("Numeric CUDA visibility does not cover the whole physical node")
        result = [physical[device] for device in devices]
    elif set(devices) == set(physical.values()):
        result = devices
    else:
        raise ValueError("CUDA UUID visibility does not cover the whole physical node")
    if len(set(result)) != allocated_count:
        raise ValueError("A GPU UUID would be assigned more than once")
    return result


def allocation():
    job_id = os.environ.get("SLURM_JOB_ID", "")
    count = os.environ.get("SLURM_GPUS_ON_NODE", "")
    if not job_id.isdigit() or not count.isdigit() or int(count) < 1:
        raise ValueError("Slurm job identity and numeric SLURM_GPUS_ON_NODE are required")
    if os.environ.get("SLURM_JOB_NUM_NODES", "1") != "1":
        raise ValueError("This dispatcher requires one complete GPU node")
    completed = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True, timeout=30)
    inventory = [{"index": int(index.strip()), "uuid": uuid.strip(), "name": name.strip(), "memory_mib": int(memory.strip())} for index, uuid, name, memory in csv.reader(io.StringIO(completed.stdout))]
    devices = canonical_devices(os.environ.get("CUDA_VISIBLE_DEVICES", ""), inventory, int(count))
    return {"job_id": job_id, "host": socket.gethostname(), "inventory": inventory, "worker_gpu_uuids": devices, "gpu_count": len(devices)}


def create_smoke(entry, directory, allocation_info, worker_index, launch_id):
    source = Path(entry["prepared_dir"])
    cell = copy.deepcopy(entry["cell"])
    rows = read_jsonl(source / "prompts.jsonl")[:1]
    if not rows:
        raise ValueError("Smoke cell has no representative prompt")
    profile_id = f"launch-smoke-{launch_id}-{worker_index}"
    cell["profile"] = profile_id
    cell["profile_config"].update(id=profile_id, samples=1)
    cell.update(samples=1, expected_problems=1, expected_responses=1)
    cell["smoke"] = {"production_cell_id": entry["cell"]["cell_id"], "launch_id": launch_id, "gpu_uuid": allocation_info["worker_gpu_uuids"][worker_index], "excluded_from_production": True}
    cell["comparison_sha256"] = digest({"production": cell["comparison_sha256"], "smoke": cell["smoke"]})
    cell.pop("cell_id")
    cell["cell_id"] = digest(cell)[:24]
    output = directory / cell["cell_id"]
    output.mkdir(parents=True)
    write_json(output / "cell.json", cell)
    (output / "prompts.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest = read_json(source / "preparation.json")
    manifest.update(source_cell_id=entry["source_cell_id"], files={name: file_hash(output / name) for name in ("cell.json", "prompts.jsonl")}, smoke=cell["smoke"])
    write_json(output / "preparation.json", manifest)
    return {"prepared_dir": str(output), "cell": cell, "source_cell_id": entry["source_cell_id"]}


def verify_complete(entry, output_root):
    directory = Path(output_root) / entry["cell"]["cell_id"]
    receipt = read_json(directory / "receipt.json")
    preparation = Path(entry["prepared_dir"]) / "preparation.json"
    if receipt.get("status") != "complete" or receipt.get("cell_id") != entry["cell"]["cell_id"]:
        raise ValueError("Cell process did not produce a complete receipt")
    if receipt["preparation_sha256"] != file_hash(preparation) or receipt["records_sha256"] != file_hash(directory / "records.jsonl") or receipt["probes_sha256"] != file_hash(directory / "probes.json"):
        raise ValueError("Completed cell receipt hashes do not match")
    records = read_jsonl(directory / "records.jsonl")
    validate_records(records, read_jsonl(Path(entry["prepared_dir"]) / "prompts.jsonl"), entry["cell"], complete=True)
    if receipt["responses"] != len(records):
        raise ValueError("Completed cell response count mismatch")
    return {"cell_id": entry["cell"]["cell_id"], "receipt_sha256": file_hash(directory / "receipt.json"), "responses": len(records)}


class Children:
    def __init__(self):
        self.lock = threading.Lock()
        self.processes = {}
        self.stopped = threading.Event()

    def stop(self):
        self.stopped.set()
        with self.lock:
            for process in list(self.processes.values()):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def execute(self, command, environment, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as stream:
            with self.lock:
                if self.stopped.is_set():
                    raise InterruptedError("Dispatcher stopped before launching this cell")
                process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                self.processes[process.pid] = process
            stopped_at = None
            try:
                while process.poll() is None:
                    if self.stopped.is_set():
                        stopped_at = stopped_at or time.monotonic()
                        if time.monotonic() - stopped_at > 30:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                    try:
                        process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        pass
                if process.returncode:
                    raise RuntimeError(f"Cell subprocess exited {process.returncode}; inspect {log_path}")
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.1)
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                with self.lock:
                    self.processes.pop(process.pid, None)


def run_process(entry, output_root, device, logs, children):
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES=device, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    command = [sys.executable, str(EVALUATION / "sweep.py"), "run", "--prepared", entry["prepared_dir"], "--output-root", str(output_root)]
    children.execute(command, environment, logs / f"{entry['cell']['cell_id']}.log")
    return verify_complete(entry, output_root)


def run(args, package_sha256):
    if not args.workers_from_visible:
        raise ValueError("--workers-from-visible is required; every allocated GPU must be assigned")
    entries, identity = prepared_entries(args, package_sha256)
    allocation_info = allocation()
    scope = args.results_root / ".dispatch" / identity["plan_sha256"][:24]
    scope.mkdir(parents=True, exist_ok=True)
    with (scope / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch_id = f"{allocation_info['job_id']}-{time.time_ns()}"
        directory = scope / launch_id
        directory.mkdir()
        receipt_path = directory / "dispatch.json"
        receipt = {"schema_version": 1, "status": "smoke", "started_at": now(), "launch_id": launch_id, **identity, "allocation": allocation_info, "completed": [], "smoke": [], "logs": str(directory / "logs")}
        write_json(receipt_path, receipt)
        children = Children()
        previous_handlers = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, lambda number, frame: children.stop())
        try:
            candidates = {entry["cell"]["model_id"]: entry for entry in entries if entry["cell"]["profile"] == "greedy-native" and entry["cell"]["benchmark"] == "math500"}
            if set(candidates) != set(MODEL_IDS):
                raise ValueError("Every model arm needs a prepared greedy MATH-500 smoke source")
            devices = allocation_info["worker_gpu_uuids"]
            smoke_entries = [create_smoke(candidates[MODEL_IDS[index % len(MODEL_IDS)]], directory / "smoke-prepared", allocation_info, index, launch_id) for index in range(len(devices))]
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
                futures = [pool.submit(run_process, entry, directory / "smoke-results", devices[index], directory / "logs" / f"gpu-{index}-smoke", children) for index, entry in enumerate(smoke_entries)]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        receipt["smoke"].append(future.result())
                        write_json(receipt_path, receipt)
                    except BaseException:
                        children.stop()
                        raise
            if children.stopped.is_set():
                raise InterruptedError("Dispatcher stopped during smoke validation")
            receipt.update(status="running", smoke_completed_at=now())
            write_json(receipt_path, receipt)
            pending = queue.Queue()
            for entry in sorted(entries, key=lambda item: item["cell"]["expected_responses"] * item["cell"]["budget"], reverse=True):
                pending.put(entry)
            receipt_lock = threading.Lock()

            def worker(index, device):
                while not children.stopped.is_set():
                    try:
                        entry = pending.get_nowait()
                    except queue.Empty:
                        return
                    try:
                        result = run_process(entry, args.results_root, device, directory / "logs" / f"gpu-{index}", children)
                        with receipt_lock:
                            receipt["completed"].append({**result, "worker": index, "gpu_uuid": device})
                            receipt["updated_at"] = now()
                            write_json(receipt_path, receipt)
                    except BaseException:
                        children.stop()
                        raise
                    finally:
                        pending.task_done()

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
                futures = [pool.submit(worker, index, device) for index, device in enumerate(devices)]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            if children.stopped.is_set() or len(receipt["completed"]) != len(entries):
                raise InterruptedError("Dispatch stopped before every production cell completed")
            receipt.update(status="complete", completed_at=now())
            write_json(receipt_path, receipt)
            return receipt
        except BaseException as error:
            children.stop()
            receipt.update(status="failed", failed_at=now(), error={"type": type(error).__name__, "message": str(error)})
            write_json(receipt_path, receipt)
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, default=ROOT)
    parser.add_argument("--workers-from-visible", action="store_true")
    args = parser.parse_args()
    package_sha256 = verify_package()
    result = prepare(args, package_sha256) if args.command == "prepare" else run(args, package_sha256)
    print(json.dumps({"status": result["status"], "cells": len(result.get("cells", result.get("completed", [])))}), flush=True)


if __name__ == "__main__":
    main()
