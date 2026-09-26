import hashlib
import json
import os
import pickle
import re
import shutil
import tempfile
from pathlib import Path

from deepseek_study.rollouts.queue import QueueState


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save(directory: Path, state: QueueState, config_hash: str, identity_hash: str, components_hash: str | None = None):
    state.validate()
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=directory, prefix="queue.pkl.")
    try:
        with os.fdopen(fd, "wb") as stream:
            pickle.dump(state, stream, protocol=5)
            stream.flush()
            os.fsync(stream.fileno())
        queue_hash = file_digest(Path(temporary))
        os.replace(temporary, directory / "queue.pkl")
        fsync_directory(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    marker = {
        "format": 2,
        "step": state.completed_steps,
        "lag": state.lag,
        "config_sha256": config_hash,
        "identity_sha256": identity_hash,
        "components_sha256": components_hash,
        "queue_sha256": queue_hash,
    }
    atomic_write(directory / "complete.json", (json.dumps(marker, indent=2) + "\n").encode())


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(directory: Path, config_hash: str, identity_hash: str):
    marker = json.loads((directory / "complete.json").read_text())
    if marker["format"] != 2 or marker["config_sha256"] != config_hash or marker["identity_sha256"] != identity_hash:
        raise ValueError("Checkpoint protocol differs from the requested study")
    if file_digest(directory / "queue.pkl") != marker["queue_sha256"]:
        raise ValueError("Checkpoint queue checksum mismatch")
    with (directory / "queue.pkl").open("rb") as stream:
        state = pickle.load(stream)
    if not isinstance(state, QueueState) or state.completed_steps != marker["step"] or state.lag != marker["lag"]:
        raise ValueError("Checkpoint state differs from its commit marker")
    state.validate()
    return state


def seal(checkpoint, trainer_gpus):
    files = []
    for relative in ("trainer", "orchestrator", "rng"):
        directory = checkpoint / relative
        paths = sorted(path for path in directory.rglob("*") if path.is_file())
        if not paths:
            raise ValueError(f"Missing checkpoint component: {relative}")
        if relative == "rng" and len(paths) != trainer_gpus:
            raise ValueError("Missing trainer RNG state")
        for path in paths:
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            record = {"path": str(path.relative_to(checkpoint)), "size": path.stat().st_size}
            if relative != "trainer" or path.name == ".metadata":
                record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append(record)
        fsync_directory(directory)
    atomic_write(checkpoint / "study" / "components.json", (json.dumps(files, indent=2) + "\n").encode())
    return hashlib.sha256((checkpoint / "study" / "components.json").read_bytes()).hexdigest()


def verify_components(checkpoint):
    path = checkpoint / "study" / "components.json"
    marker = json.loads((checkpoint / "study" / "complete.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != marker.get("components_sha256"):
        raise ValueError("Checkpoint component manifest checksum mismatch")
    for record in json.loads(path.read_text()):
        target = (checkpoint / record["path"]).resolve()
        if (
            not target.is_relative_to(checkpoint.resolve())
            or not target.is_file()
            or target.stat().st_size != record["size"]
        ):
            raise ValueError("Checkpoint component is missing or incomplete")
        if "sha256" in record and hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Checkpoint component checksum mismatch")


def prune_complete(root, keep_last, keep_interval, identity_hash):
    completed = []
    for directory in root.iterdir():
        if directory.is_symlink() or not directory.is_dir() or not re.fullmatch(r"step_\d+", directory.name):
            continue
        marker = directory / "study" / "complete.json"
        if marker.is_file():
            record = json.loads(marker.read_text())
            step = int(directory.name.removeprefix("step_"))
            if record.get("identity_sha256") != identity_hash or record.get("step") != step:
                raise ValueError("Refusing to prune checkpoints from a different run identity")
            completed.append((step, directory))
    completed.sort()
    retained = {step for step, _ in completed[-keep_last:]}
    removed = []
    for step, directory in completed:
        if step not in retained and step % keep_interval != 0:
            shutil.rmtree(directory)
            removed.append(step)
    fsync_directory(root)
    return removed
