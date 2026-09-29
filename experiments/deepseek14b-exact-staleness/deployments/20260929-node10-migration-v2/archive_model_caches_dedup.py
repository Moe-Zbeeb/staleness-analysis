import fcntl
import hashlib
import json
import math
import os
import shutil
import socket
import stat
import tempfile
import time
from pathlib import Path

ROOT = Path("/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/archives/node10-model-caches-20260929")
SOURCE_ROOT = Path("/tmp")
REQUIRED_UID = 29562
REQUIRED_NODE = "deep-chungus-10"
EXPECTED = {
    "2142210": "Qwen3-14B-40c06982",
    "2142188": "Qwen3-14B-40c06982",
    "2142052": "Qwen2.5-Math-7B-b101308f",
    "2142033": "Qwen2.5-Math-7B-b101308f",
    "2142029": "Qwen2.5-Math-7B-b101308f",
    "2142032": "Qwen2.5-3B-3aab1f19",
    "2142027": "Qwen2.5-3B-3aab1f19",
}


def owned(path, directory=False):
    path = Path(path)
    info = path.lstat()
    if info.st_uid != REQUIRED_UID or stat.S_ISLNK(info.st_mode):
        raise ValueError(f"Unexpected owner or symlink: {path}")
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError(f"Unexpected file type: {path}")
    return info


def stamp(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def relative(name):
    path = Path(name)
    if not isinstance(name, str) or path.is_absolute() or ".." in path.parts or str(path) != name or name in {"", "."}:
        raise ValueError("Unsafe file manifest path")
    return path


def digest(path):
    initial = owned(path)
    result = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if stamp(os.fstat(stream.fileno())) != stamp(initial):
            raise ValueError(f"File replaced before hashing: {path}")
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
        if stamp(os.fstat(stream.fileno())) != stamp(initial):
            raise ValueError(f"File changed during hashing: {path}")
    if stamp(owned(path)) != stamp(initial):
        raise ValueError(f"File changed after hashing: {path}")
    return result.hexdigest()


def inventory(root, hashes):
    root = Path(root)
    root_stat = owned(root, directory=True)
    files, snapshots = {}, {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            owned(path, directory=True)
            continue
        owned(path)
        name = str(path.relative_to(root))
        relative(name)
        files[name] = {"size": info.st_size}
        snapshots[name] = stamp(info)
        if hashes:
            files[name]["sha256"] = digest(path)
    if stamp(owned(root, directory=True)) != stamp(root_stat):
        raise ValueError(f"Directory changed during inventory: {root}")
    return files, snapshots


def manifest_id(files):
    if not isinstance(files, dict) or not files or not any(name.endswith(".safetensors") for name in files):
        raise ValueError("Archive manifest has no model weights")
    for name, record in files.items():
        relative(name)
        if not isinstance(record, dict) or set(record) != {"size", "sha256"}:
            raise ValueError("Invalid archive manifest record")
        if type(record["size"]) is not int or record["size"] < 0:
            raise ValueError("Invalid archive manifest size")
        checksum = record["sha256"]
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("Invalid archive manifest checksum")
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_for(job):
    if job not in EXPECTED:
        raise ValueError("Job is outside the cache archive allowlist")
    return SOURCE_ROOT / ("prime-rl-" + job) / EXPECTED[job]


def receipt_for(job):
    return ROOT / ("prime-rl-" + job) / "verified.json"


def ensure_directory(path, boundary):
    path, boundary = Path(path), Path(boundary)
    if not path.is_relative_to(boundary):
        raise ValueError("Directory escaped its archive boundary")
    owned(boundary, directory=True)
    current = boundary
    for name in path.relative_to(boundary).parts:
        current = current / name
        if not current.exists() and not current.is_symlink():
            current.mkdir()
        owned(current, directory=True)


def allowed_archive(path, fingerprint):
    path = Path(path)
    old = {ROOT / ("prime-rl-" + job) / model for job, model in EXPECTED.items()}
    new = {ROOT / "canonical" / fingerprint / model for model in EXPECTED.values()}
    if path not in old | new:
        raise ValueError("Receipt archive path is outside the exact allowlist")
    current = ROOT
    owned(current, directory=True)
    for name in path.relative_to(ROOT).parts:
        current = current / name
        owned(current, directory=True)
    return path


def load_verified(job, chain=()):
    if job in chain:
        raise ValueError("Canonical receipt references form a cycle")
    receipt_path = receipt_for(job)
    owned(receipt_path.parent, directory=True)
    initial = owned(receipt_path)
    raw = receipt_path.read_bytes()
    if stamp(owned(receipt_path)) != stamp(initial):
        raise ValueError("Archive receipt changed while reading")
    body = json.loads(raw)
    if body.get("source") != str(source_for(job)):
        raise ValueError("Archive receipt source does not match its allowed job")
    fingerprint = manifest_id(body.get("files"))
    if body.get("manifest_sha256", fingerprint) != fingerprint:
        raise ValueError("Archive receipt manifest checksum mismatch")
    verified_at = body.get("verified_at")
    if type(verified_at) not in (int, float) or not math.isfinite(verified_at) or not 0 < verified_at <= time.time():
        raise ValueError("Archive receipt has no valid completed verification time")
    archive = allowed_archive(body.get("archive", ""), fingerprint)
    limit = verified_at
    if body.get("format") == 2:
        anchor = body.get("canonical_receipt")
        allowed_receipts = {str(receipt_for(candidate)): candidate for candidate in EXPECTED}
        if anchor not in allowed_receipts:
            raise ValueError("Canonical receipt pointer is outside the exact allowlist")
        anchor_job = allowed_receipts[anchor]
        original_time = body.get("canonical_verified_at")
        if (
            type(original_time) not in (int, float)
            or not math.isfinite(original_time)
            or not 0 < original_time <= verified_at
        ):
            raise ValueError("Canonical verification time is invalid")
        if anchor_job != job:
            original = load_verified(anchor_job, (*chain, job))
            if (
                original["files"] != body["files"]
                or original["archive"] != str(archive)
                or original_time != original["verified_at"]
            ):
                raise ValueError("Reused archive differs from its canonical verification proof")
        limit = original_time
    elif "format" in body:
        raise ValueError("Unknown archive verification receipt format")
    actual, _ = inventory(archive, hashes=False)
    if actual != {name: {"size": item["size"]} for name, item in body["files"].items()}:
        raise ValueError("Verified archive inventory or size changed")
    if any((archive / name).stat().st_mtime_ns > int(limit * 1_000_000_000) for name in actual):
        raise ValueError("Verified archive content changed after verification")
    return {
        **body,
        "manifest_sha256": fingerprint,
        "receipt_path": str(receipt_path),
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
    }


def verify_environment():
    if os.getuid() != REQUIRED_UID:
        raise ValueError("Archive must run as the authorized owner")
    if os.environ.get("SLURMD_NODENAME") != REQUIRED_NODE or socket.gethostname().split(".")[0] != REQUIRED_NODE:
        raise ValueError("Archive must run on the authorized node")
    if not Path("/proc").is_dir():
        raise RuntimeError("Process ownership checks are unavailable")
    owned(ROOT, directory=True)
    for job in EXPECTED:
        parent = source_for(job).parent
        if parent.exists() or parent.is_symlink():
            owned(parent, directory=True)
    check_processes()


def check_processes(source=None):
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != REQUIRED_UID:
                continue
            command = (proc / "cmdline").read_bytes()
            if b"archive_old_model_caches.py" in command:
                raise RuntimeError("Original archive worker is still running: " + proc.name)
            if source is not None and (str(source).encode() in command or str(source.parent).encode() in command):
                raise RuntimeError("Old cache is referenced by a running process: " + proc.name)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass


def sync_dir(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_receipt(job, body):
    target = receipt_for(job)
    ensure_directory(target.parent, ROOT)
    descriptor, name = tempfile.mkstemp(dir=target.parent, prefix=".verified-")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(body, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        sync_dir(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def copy_model(source, destination, expected, source_stamp):
    ensure_directory(destination, ROOT)
    partial, _ = inventory(destination, hashes=False)
    if set(partial) - set(expected):
        raise ValueError("Unverified archive contains unexpected files; source retained")
    for name in expected:
        incoming = source / relative(name)
        outgoing = destination / name
        ensure_directory(outgoing.parent, ROOT)
        if outgoing.exists() or outgoing.is_symlink():
            owned(outgoing)
        descriptor, temporary_name = tempfile.mkstemp(dir=outgoing.parent, prefix=".archive-copy-")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream, incoming.open("rb") as original:
                shutil.copyfileobj(original, stream, length=8 * 1024 * 1024)
                stream.flush()
                os.fsync(stream.fileno())
            shutil.copystat(incoming, temporary, follow_symlinks=False)
            os.replace(temporary, outgoing)
            sync_dir(outgoing.parent)
        finally:
            temporary.unlink(missing_ok=True)
    after, _ = inventory(destination, hashes=True)
    current, current_stamp = inventory(source, hashes=False)
    if (
        after != expected
        or current != {name: {"size": item["size"]} for name, item in expected.items()}
        or current_stamp != source_stamp
    ):
        raise RuntimeError("Archive verification failed; original cache retained")
    sync_dir(destination)


def reclaim(job, verified, expected, source_stamp):
    source = source_for(job)
    check_processes(source)
    current, current_stamp = inventory(source, hashes=False)
    if current != {name: {"size": item["size"]} for name, item in expected.items()} or current_stamp != source_stamp:
        raise RuntimeError("Original cache changed before reclamation")
    checked = load_verified(job)
    if checked["files"] != expected or checked["archive"] != verified["archive"]:
        raise ValueError("Published archive proof differs from the source")
    retired = source.with_name(source.name + ".verified-archive-20260929")
    if retired.exists() or retired.is_symlink():
        raise FileExistsError("Previous cache reclamation is unfinished; inspect before continuing")
    source.rename(retired)
    shutil.rmtree(retired)
    sync_dir(source.parent)


def run():
    verify_environment()
    lock_path = ROOT / ".dedup-archive.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        owned(lock_path)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_processes()
        registry = {}
        receipts = {}
        for job in EXPECTED:
            receipt_path = receipt_for(job)
            if receipt_path.exists() or receipt_path.is_symlink():
                receipt = load_verified(job)
                receipts[job] = receipt
                registry.setdefault(receipt["manifest_sha256"], receipt)
        results = []
        for job, model in EXPECTED.items():
            source = source_for(job)
            if not source.exists() and not source.is_symlink():
                if job not in receipts:
                    raise ValueError("Absent original cache has no verified archive receipt: " + job)
                if (
                    source.with_name(source.name + ".verified-archive-20260929").exists()
                    or source.with_name(source.name + ".verified-archive-20260929").is_symlink()
                ):
                    raise RuntimeError("Original cache reclamation is unfinished: " + job)
                results.append({"job": job, "status": "already_archived", "archive": receipts[job]["archive"]})
                continue
            check_processes(source)
            files, source_stamp = inventory(source, hashes=True)
            fingerprint = manifest_id(files)
            if job in receipts:
                canonical = receipts[job]
                if canonical["files"] != files:
                    raise ValueError("Source cache differs from its already verified archive: " + job)
            else:
                canonical = registry.get(fingerprint)
                reused = canonical is not None
                if canonical is not None:
                    canonical = load_verified(Path(canonical["receipt_path"]).parent.name.removeprefix("prime-rl-"))
                    if canonical["files"] != files:
                        raise ValueError("Canonical model manifest mismatch")
                else:
                    old_destination = ROOT / ("prime-rl-" + job) / model
                    destination = (
                        old_destination
                        if old_destination.exists() or old_destination.is_symlink()
                        else ROOT / "canonical" / fingerprint / model
                    )
                    print(
                        json.dumps(
                            {
                                "phase": "copying_unique_model",
                                "job": job,
                                "archive": str(destination),
                                "bytes": sum(row["size"] for row in files.values()),
                            }
                        ),
                        flush=True,
                    )
                    copy_model(source, destination, files, source_stamp)
                    canonical = {"archive": str(destination), "files": files, "verified_at": time.time()}
                body = {
                    "format": 2,
                    "source": str(source),
                    "archive": canonical["archive"],
                    "files": files,
                    "verified_at": time.time(),
                    "manifest_sha256": fingerprint,
                    "reused_verified_archive": reused,
                    "canonical_receipt": canonical.get("receipt_path", str(receipt_for(job))),
                    "canonical_verified_at": canonical["verified_at"],
                }
                publish_receipt(job, body)
                canonical = load_verified(job)
                registry.setdefault(fingerprint, canonical)
            reclaim(job, canonical, files, source_stamp)
            result = {
                "job": job,
                "status": "archived_and_reclaimed",
                "archive": canonical["archive"],
                "manifest_sha256": fingerprint,
                "free_bytes": shutil.disk_usage(SOURCE_ROOT).free,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
        return results
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    print(json.dumps({"status": "complete", "results": run()}, indent=2), flush=True)
