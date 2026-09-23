import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import native_gpu


PACKAGE = Path(__file__).resolve().parent
FROZEN_MANIFEST_SHA256 = "244f3e25f6dcd6108e93cb502251caf87e3cd2183e1f5088c7089362783b8f61"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_package(root=PACKAGE):
    manifest_path = root / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or not {"resume.py", "evaluate_job.sh", "native_gpu.py", "native_devices.py"} <= manifest.get("files", {}).keys():
        raise ValueError("Recovery manifest does not cover runtime files")
    for name, expected in manifest["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or file_hash(root / name) != expected:
            raise ValueError(f"Recovery source hash mismatch: {name}")
    return file_hash(manifest_path)


def validate_idle_allocation(info):
    if info.get("host", "").split(".")[0] != "deep-chungus-11" or info.get("hardware_cohort") != "a100-80gb":
        raise ValueError("This recovery preserves the node11 A10080GB comparison cohort")
    selected = info.get("worker_gpu_uuids", [])
    inventory = info.get("inventory", [])
    if info.get("gpu_count") != 2 or len(set(selected)) != 2 or len(inventory) != 2 or {item.get("uuid") for item in inventory} != set(selected):
        raise ValueError("Recovery requires exactly two fully inventoried allocated GPU UUIDs")
    for item in inventory:
        if "compute_processes" not in item or not isinstance(item["compute_processes"], list):
            raise ValueError("Allocated GPU compute-process evidence is missing")
        if item["compute_processes"]:
            pids = [process.get("pid") for process in item["compute_processes"]]
            raise ValueError(f"Allocated GPU {item['uuid']} already has compute processes: {pids}")
    return info


def gated_allocation(args, original_allocation, package_sha256, source):
    info = original_allocation(args)
    provenance = {
        "source_manifest_sha256": package_sha256,
        "frozen_launcher_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "policy": "Preserve native Slurm CUDA assignment; reject every pre-existing compute process on the selected UUIDs",
    }
    directory = args.results_root / ".native-recovery" / f"{info['job_id']}-{info['restart_count']}"
    directory.mkdir(parents=True, exist_ok=True)
    receipt = {"schema_version": 1, "status": "checking", "allocation": info, "occupancy_guard": provenance}
    receipt_path = directory / "allocation.json"
    if receipt_path.exists():
        raise ValueError("This job/restart already has an occupancy recovery receipt")
    source.write_json(receipt_path, receipt)
    try:
        validate_idle_allocation(info)
    except BaseException as error:
        receipt.update(status="blocked", error={"type": type(error).__name__, "message": str(error)})
        source.write_json(receipt_path, receipt)
        raise
    receipt["status"] = "passed"
    source.write_json(receipt_path, receipt)
    info["occupancy_guard"] = {**provenance, "receipt_path": str(receipt_path), "receipt_sha256": file_hash(receipt_path)}
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-launcher", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--derived-prepared-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--original-results-root", type=Path)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--gres-config", type=Path, required=True)
    parser.add_argument("--expected-gres-sha256", required=True)
    args = parser.parse_args()
    args.workers_from_visible = True
    args.shard = 1
    args.shards = 2
    package_sha256 = verify_package()
    frozen = args.frozen_launcher.resolve()
    if file_hash(frozen / "source-manifest.json") != FROZEN_MANIFEST_SHA256:
        raise ValueError("Frozen v2 launcher manifest changed")
    sys.path.insert(0, str(frozen))
    spec = importlib.util.spec_from_file_location("frozen_partial_dispatch", frozen / "newdispatcher.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    launcher_sha256 = module.verify_package()
    if launcher_sha256 != FROZEN_MANIFEST_SHA256:
        raise ValueError("Frozen launcher verification disagrees with its pinned manifest")
    source, source_sha256 = module.load_source(args.source_root)
    import gpu_devices
    if Path(gpu_devices.__file__).resolve() != frozen / "gpu_devices.py":
        raise ValueError("GPU mapper origin differs from the frozen launcher")
    original_allocation = gpu_devices.allocation
    gpu_devices.allocation = lambda options: gated_allocation(options, lambda opts: native_gpu.allocation(opts, gpu_devices), package_sha256, source)
    try:
        result = module.run(args, source, source_sha256, launcher_sha256)
    finally:
        gpu_devices.allocation = original_allocation
    print(json.dumps({"status": result["status"], "shard": 1, "cells": len(result["completed"]), "occupancy_guard_sha256": package_sha256}), flush=True)


if __name__ == "__main__":
    main()
