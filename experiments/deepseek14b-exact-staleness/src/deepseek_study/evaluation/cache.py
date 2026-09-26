import contextlib
import fcntl
import hashlib
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from .common import io_slot, lock, read, write
from .export import validate_export


def prune(directory, keep=2):
    models = sorted((p for p in directory.iterdir() if p.is_dir() and (p / "evaluation-ready.json").is_file()),
                    key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for model in models[keep:]:
        try:
            with lock(directory / f"{model.name}.use.lock", blocking=False):
                shutil.rmtree(model)
        except BlockingIOError:
            continue


@contextlib.contextmanager
def staged_model(pool, source, checkpoint_id, directory=None, verify_filesystem=True):
    if len(checkpoint_id) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_id):
        raise ValueError("Invalid checkpoint cache identity")
    directory = Path(directory or f"/tmp/deepseek-evaluation-{os.getuid()}")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink() or directory.stat().st_uid != os.getuid():
        raise ValueError("Evaluator cache must be an owned, non-symlink directory")
    if verify_filesystem:
        filesystem = subprocess.run(["findmnt", "-T", str(directory), "-n", "-o", "FSTYPE"],
                                    check=True, text=True, capture_output=True).stdout.strip()
        if filesystem not in {"ext4", "xfs", "btrfs", "zfs", "overlay"}:
            raise ValueError(f"Model cache must use local disk, not {filesystem}")
    destination = directory / checkpoint_id
    with lock(directory / "cache.lock"):
        for orphan in directory.glob(".partial-*"):
            if orphan.is_dir() and not orphan.is_symlink():
                shutil.rmtree(orphan)
        if not destination.exists():
            prune(directory, keep=1)
            receipt = validate_export(source, checkpoint_id, hashes=False)
            size = sum(item["size"] for item in receipt["files"].values())
            if shutil.disk_usage(directory).free < size + 5 * 1024**3:
                raise OSError("Not enough local disk for this checkpoint plus 5 GiB headroom")
            temporary = directory / (".partial-" + uuid.uuid4().hex)
            temporary.mkdir()
            try:
                with io_slot(pool):
                    for name, record in receipt["files"].items():
                        if Path(name).name != name:
                            raise ValueError("Unexpected model export path")
                        checksum = hashlib.sha256()
                        with (Path(source) / name).open("rb") as input_stream, (temporary / name).open("wb") as output:
                            while block := input_stream.read(8 * 1024**2):
                                checksum.update(block)
                                output.write(block)
                            output.flush()
                            os.fsync(output.fileno())
                        if checksum.hexdigest() != record["sha256"]:
                            raise ValueError("Source model checksum changed during staging")
                write(temporary / "evaluation-ready.json", receipt)
                os.rename(temporary, destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        else:
            validate_export(destination, checkpoint_id)
        os.utime(destination, None)
        usage = (directory / f"{checkpoint_id}.use.lock").open("a+")
        fcntl.flock(usage, fcntl.LOCK_SH)
    try:
        yield destination, read(destination / "evaluation-ready.json")
    finally:
        usage.close()
        with lock(directory / "cache.lock"):
            prune(directory)
