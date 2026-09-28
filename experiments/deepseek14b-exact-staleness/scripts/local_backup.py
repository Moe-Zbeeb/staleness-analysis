import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import time
import uuid
from pathlib import Path


METRIC_NAMES = {
    "metrics.jsonl",
    "updates.jsonl",
    "generations.jsonl",
    "grading.jsonl",
    "paper-metrics.jsonl",
    "evaluation-metrics.jsonl",
    "run.json",
    "run-status.json",
    "study-complete.json",
    "preflight.json",
    "deployment.json",
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verified_copy(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Backup input must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("Refusing a symlink backup destination")
    initial = source.stat()
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    expected = hashlib.sha256()
    remaining = initial.st_size
    try:
        with source.open("rb") as incoming, temporary.open("xb") as outgoing:
            while remaining:
                block = incoming.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError("Backup input shrank during copying")
                expected.update(block)
                outgoing.write(block)
                remaining -= len(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        final = source.stat()
        if final.st_ino != initial.st_ino or final.st_size < initial.st_size:
            raise ValueError("Backup input was replaced or truncated")
        if final.st_size == initial.st_size and final.st_mtime_ns != initial.st_mtime_ns:
            raise ValueError("Backup input changed in place")
        if digest(temporary) != expected.hexdigest():
            raise ValueError("Shared backup checksum mismatch")
        os.replace(temporary, destination)
        sync_directory(destination.parent)
        return {"bytes": initial.st_size, "sha256": expected.hexdigest(), "mtime_ns": initial.st_mtime_ns}
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_files(source):
    marker = source / "study/complete.json"
    if not marker.is_file():
        return None
    body = json.loads(marker.read_text())
    components = source / "study/components.json"
    if digest(components) != body["components_sha256"]:
        raise ValueError("Invalid local checkpoint component manifest")
    if digest(source / "study/queue.pkl") != body["queue_sha256"]:
        raise ValueError("Invalid local checkpoint queue")
    records = json.loads(components.read_text())
    for record in records:
        path = source / record["path"]
        if not path.resolve().is_relative_to(source.resolve()) or path.is_symlink():
            raise ValueError("Checkpoint component escapes its directory")
        if path.stat().st_size != record["size"]:
            raise ValueError("Checkpoint component has the wrong size")
        if record.get("sha256") and digest(path) != record["sha256"]:
            raise ValueError("Checkpoint component has the wrong checksum")
    return body


def backup_checkpoint(source, destination, owner):
    source, destination = Path(source), Path(destination)
    commit = source / "study/complete.json"
    if not commit.is_file():
        return False
    marker = json.loads(commit.read_text())
    if source.name != f"step_{marker['step']}":
        raise ValueError("Checkpoint directory and policy clock disagree")
    if marker["identity_sha256"] != owner["identity_sha256"] or marker["config_sha256"] != owner["config_sha256"]:
        raise ValueError("Checkpoint belongs to another run")
    if destination.exists():
        receipt = json.loads((destination / "backup-verified.json").read_text())
        if receipt["run_uuid"] != owner["run_uuid"] or receipt["checkpoint"] != marker:
            raise ValueError("Shared checkpoint belongs to another run")
        return False
    checkpoint_files(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / (".incoming-" + destination.name + "-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        files = {}
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError("Checkpoint contains a symlink")
            if path.is_file():
                relative = str(path.relative_to(source))
                files[relative] = verified_copy(path, staging / relative)
        if json.loads((source / "study/complete.json").read_text()) != marker:
            raise ValueError("Checkpoint changed during backup")
        checkpoint_files(staging)
        atomic_json(
            staging / "backup-verified.json",
            {
                "run_uuid": owner["run_uuid"],
                "checkpoint": marker,
                "files": files,
                "verified_at": time.time(),
            },
        )
        os.rename(staging, destination)
        sync_directory(destination.parent)
        return True
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def prune_verified_checkpoints(root, owner, keep_last=2, keep_interval=100):
    completed = []
    for path in Path(root).glob("step_*"):
        receipt_path = path / "backup-verified.json"
        if path.is_symlink() or not path.is_dir() or not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text())
        marker = json.loads((path / "study/complete.json").read_text())
        if (
            receipt["run_uuid"] != owner["run_uuid"]
            or receipt["checkpoint"] != marker
            or marker["identity_sha256"] != owner["identity_sha256"]
            or marker["config_sha256"] != owner["config_sha256"]
            or path.name != f"step_{marker['step']}"
        ):
            raise ValueError("Refusing retention across checkpoint owners")
        completed.append((marker["step"], path))
    completed.sort()
    retained = {step for step, _ in completed[-keep_last:]}
    removed = []
    for step, path in completed:
        if step not in retained and step % keep_interval:
            shutil.rmtree(path)
            removed.append(step)
    if removed:
        sync_directory(root)
    return removed


class Backup:
    def __init__(self, source, destination, metrics=None):
        self.source, self.destination = Path(source).resolve(), Path(destination).resolve()
        self.metrics = Path(metrics).resolve() if metrics else None
        roots = [self.source, self.destination, *([self.metrics] if self.metrics else [])]
        if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(roots) for b in roots[i + 1 :]):
            raise ValueError("Backup and working directories must not overlap")
        self.copied = {}
        self.owner = None

    def reserve(self, destination):
        marker = destination / "backup-owner.json"
        identity = {key: self.owner[key] for key in ("run_uuid", "config_sha256", "identity_sha256")}
        identity["source"] = str(self.source)
        if destination.exists():
            if not marker.is_file() or json.loads(marker.read_text()) != identity:
                raise ValueError("Shared backup directory is owned by another run")
        else:
            destination.mkdir(parents=True, exist_ok=False)
            atomic_json(marker, identity)

    def poll(self):
        run = self.source / "run.json"
        if not run.is_file():
            return {"status": "waiting_for_run"}
        self.owner = json.loads(run.read_text())
        self.reserve(self.destination)
        if self.metrics:
            self.reserve(self.metrics)
        checkpoints = self.source / "checkpoints"
        completed = []
        candidates = sorted(
            (p for p in checkpoints.glob("step_*") if (p / "study/complete.json").is_file()),
            key=lambda p: int(p.name.removeprefix("step_")),
        )
        selected = set(candidates[-2:])
        for path in candidates:
            if path not in selected and int(path.name.removeprefix("step_")) % 100:
                continue
            if path.is_symlink():
                raise ValueError("Checkpoint directory must not be a symlink")
            if backup_checkpoint(path, self.destination / "checkpoints" / path.name, self.owner):
                completed.append(path.name)
        removed = prune_verified_checkpoints(self.destination / "checkpoints", self.owner) if completed else []
        for path in sorted(self.source.rglob("*")):
            relative = path.relative_to(self.source)
            if relative.parts[0] in {"checkpoints", ".backup"} or "__pycache__" in relative.parts:
                continue
            if path.is_symlink():
                raise ValueError(f"Active run contains an unexpected symlink: {relative}")
            if not path.is_file() or path.name.endswith((".tmp", ".lock")):
                continue
            stat = path.stat()
            version = (stat.st_size, stat.st_mtime_ns)
            targets = [self.destination]
            if self.metrics and (
                relative.parts[0] in {"tensorboard", "tracking", "configs", "source"} or str(relative) in METRIC_NAMES
            ):
                targets.append(self.metrics)
            for target in targets:
                key = str(target / relative)
                if self.copied.get(key, {}).get("version") == version:
                    continue
                record = verified_copy(path, target / relative)
                self.copied[key] = {**record, "version": (record["bytes"], record["mtime_ns"])}
        for target in [self.destination, *([self.metrics] if self.metrics else [])]:
            atomic_json(
                target / "backup-inventory.json",
                {
                    "run_uuid": self.owner["run_uuid"],
                    "verified_at": time.time(),
                    "files": {
                        str(Path(k).relative_to(target)): v
                        for k, v in self.copied.items()
                        if Path(k).is_relative_to(target)
                    },
                },
            )
        return {
            "status": "verified",
            "verified_at": time.time(),
            "new_checkpoints": completed,
            "pruned_checkpoints": removed,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    backup = Backup(args.source, args.destination, args.metrics)
    signal.signal(signal.SIGTERM, lambda *_: args.stop_file.touch())
    with args.lock.open("a") as lock:
        while True:
            final = args.stop_file.exists()
            try:
                fcntl.flock(lock, fcntl.LOCK_EX)
                status = backup.poll()
                status["final"] = final
                atomic_json(args.status, status)
            except Exception as error:
                atomic_json(args.status, {"status": "error", "at": time.time(), "error": repr(error), "final": final})
                print(repr(error), flush=True)
                if final:
                    raise
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
            if final:
                return
            for _ in range(args.interval):
                if args.stop_file.exists():
                    break
                time.sleep(1)


if __name__ == "__main__":
    main()
