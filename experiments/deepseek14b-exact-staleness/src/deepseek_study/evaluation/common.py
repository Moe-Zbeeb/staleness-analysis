import contextlib
import fcntl
import hashlib
import json
import time
from pathlib import Path

from deepseek_study.runtime.checkpoints import atomic_write


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    atomic_write(Path(path), (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode())


@contextlib.contextmanager
def lock(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def immutable_json(path, value):
    path = Path(path)
    with lock(path.with_suffix(path.suffix + ".lock")):
        if path.exists():
            if read(path) != value:
                raise ValueError(f"Immutable record changed: {path}")
        else:
            write(path, value)


def tokenizer_identity(path):
    path = Path(path)
    names = ("config.json", "tokenizer.json", "tokenizer_config.json")
    return {name: file_hash(path / name) for name in names}


@contextlib.contextmanager
def io_slot(root, slots=2):
    """Bound concurrent large shared-filesystem reads across this owner's pool."""
    with contextlib.ExitStack() as stack:
        while True:
            for index in range(slots):
                try:
                    stack.enter_context(lock(Path(root) / "io-locks" / str(index), blocking=False))
                except BlockingIOError:
                    continue
                yield
                return
            time.sleep(1)
