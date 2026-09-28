import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import uuid


FORMAT = 1
JOB_FILE = "job.json"
RESULT_FILE = "result.json"
ARCHIVE_FILE = "result.msgpack"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _safe_path(path):
    path = Path(os.path.abspath(path))
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError(f"Symlinks are not allowed in historical rollout storage: {candidate}")
    return path


def _relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Invalid weight manifest path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("Weight manifest path must stay inside its export directory")
    if not (path.name.endswith(".safetensors") or path.name == "model.safetensors.index.json"):
        raise ValueError("Historical exports contain only safetensors weights and their index")
    return path


def _metadata(value):
    if not isinstance(value, dict):
        raise ValueError("Historical jobs require scientific metadata")
    for key in ("run_uuid", "config_sha256"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"Missing historical job identity: {key}")
    return json.loads(canonical(value))


def _job_document(job):
    if not isinstance(job, dict) or set(job) != {"format", "job_id", "metadata", "weights"} or job["format"] != FORMAT:
        raise ValueError("Unsupported historical job document")
    metadata = _metadata(job["metadata"])
    records = job["weights"]
    if not isinstance(records, list) or not records:
        raise ValueError("Historical job has no weight files")
    seen = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
            raise ValueError("Invalid weight manifest entry")
        name = str(_relative_path(record["path"]))
        digest = record["sha256"]
        if name in seen or type(record["size"]) is not int or record["size"] <= 0:
            raise ValueError("Duplicate or empty historical weight file")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid weight checksum")
        seen.add(name)
    if records != sorted(records, key=lambda row: row["path"]) or not any(
        row["path"].endswith(".safetensors") for row in records
    ):
        raise ValueError("Historical weights must be canonically ordered and include safetensors")
    body = {"format": FORMAT, "metadata": metadata, "weights": records}
    if job["job_id"] != hashlib.sha256(canonical(body)).hexdigest():
        raise ValueError("Historical job identity does not match its metadata and weights")
    return json.loads(canonical(job))


def _read_json(path):
    path = _safe_path(path)
    with path.open("rb") as stream:
        return json.load(stream)


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path, data):
    path = _safe_path(path)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _all_files(directory):
    directory = _safe_path(directory)
    if not directory.is_dir():
        raise ValueError("Historical weight source must be a directory")
    paths = []
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Historical weight directories must not contain symlinks")
        mode = path.stat().st_mode
        if stat.S_ISREG(mode):
            paths.append(path)
        elif not stat.S_ISDIR(mode):
            raise ValueError("Historical weight directories contain an unsupported file type")
    return sorted(paths, key=lambda path: path.relative_to(directory).as_posix())


def _copy_file(source, destination, expected=None):
    source, destination = _safe_path(source), _safe_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(descriptor, "rb") as incoming, destination.open("xb") as outgoing:
        before = os.fstat(incoming.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Historical weight source is not a regular file")
        while block := incoming.read(8 * 1024 * 1024):
            digest.update(block)
            size += len(block)
            outgoing.write(block)
        after = os.fstat(incoming.fileno())
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("Historical weight source changed while it was being frozen")
        outgoing.flush()
        os.fsync(outgoing.fileno())
    result = {"size": size, "sha256": digest.hexdigest()}
    if size == 0 or (expected is not None and result != {key: expected[key] for key in result}):
        raise ValueError("Historical weight copy failed checksum or size validation")
    return result


def _pin_file(source, destination, expected):
    source, destination = _safe_path(source), _safe_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError as error:
        if error.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP}:
            raise
        return _copy_file(source, destination, expected)
    if destination.is_symlink() or not stat.S_ISREG(destination.stat().st_mode):
        raise ValueError("Checkpoint weight link does not reference a regular immutable shard")
    if destination.stat().st_size != expected["size"] or sha256(destination) != expected["sha256"]:
        raise ValueError("Checkpoint weight link failed checksum validation")
    _fsync_directory(destination.parent)
    return {key: expected[key] for key in ("size", "sha256")}


def _check_index(directory, records):
    names = {record["path"] for record in records}
    for name in names:
        if PurePosixPath(name).name != "model.safetensors.index.json":
            continue
        mapping = _read_json(directory / name).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Safetensors index has no weight mapping")
        for shard in mapping.values():
            relative = _relative_path(shard)
            target = (PurePosixPath(name).parent / relative).as_posix()
            if not relative.name.endswith(".safetensors") or target not in names:
                raise ValueError("Safetensors index refers to an absent weight shard")


def _match_job(directory, job):
    directory = _safe_path(directory)
    expected = _job_document(job)
    actual = _job_document(_read_json(directory / JOB_FILE))
    if actual != expected:
        raise ValueError("Historical job belongs to a different run, configuration, or policy version")
    return expected


def validate_job(directory, job):
    directory = _safe_path(directory)
    job = _match_job(directory, job)
    weights = directory / "weights"
    files = _all_files(weights)
    actual = {path.relative_to(weights).as_posix() for path in files}
    if actual != {row["path"] for row in job["weights"]}:
        raise ValueError("Historical weight files differ from their manifest")
    for row in job["weights"]:
        path = weights / row["path"]
        if path.stat().st_size != row["size"] or sha256(path) != row["sha256"]:
            raise ValueError("Historical weight checksum mismatch")
    _check_index(weights, job["weights"])
    return job


def _publish_directory(staging, destination, job):
    if destination.exists():
        validate_job(destination, job)
        return destination
    try:
        staging.rename(destination)
    except FileExistsError:
        validate_job(destination, job)
    _fsync_directory(destination.parent)
    return destination


def freeze_export(source, destination, metadata):
    source, destination = _safe_path(source), _safe_path(destination)
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Frozen export must be outside its source weight directory")
    metadata = _metadata(metadata)
    files = [
        path
        for path in _all_files(source)
        if path.name.endswith(".safetensors") or path.name == "model.safetensors.index.json"
    ]
    if not any(path.name.endswith(".safetensors") for path in files):
        raise ValueError("Historical export contains no safetensors weights")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as scratch:
        staging = Path(scratch) / "export"
        staging.mkdir()
        records = []
        for path in files:
            name = path.relative_to(source).as_posix()
            _relative_path(name)
            records.append({"path": name, **_copy_file(path, staging / "weights" / name)})
        _check_index(staging / "weights", records)
        body = {"format": FORMAT, "metadata": metadata, "weights": records}
        job = {**body, "job_id": hashlib.sha256(canonical(body)).hexdigest()}
        _atomic_write(staging / JOB_FILE, canonical(job))
        _publish_directory(staging, destination, job)
    return job


def _copy_job(source, destination, job, pin=False):
    source, destination = _safe_path(source), _safe_path(destination)
    job = validate_job(source, job)
    if destination.exists():
        validate_job(destination, job)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as scratch:
        staging = Path(scratch) / "export"
        staging.mkdir()
        copy = _pin_file if pin else _copy_file
        for row in job["weights"]:
            copy(source / "weights" / row["path"], staging / "weights" / row["path"], row)
        _atomic_write(staging / JOB_FILE, canonical(job))
        _publish_directory(staging, destination, job)
    return destination


def publish_job(localdirectory, sharedroot):
    localdirectory, sharedroot = _safe_path(localdirectory), _safe_path(sharedroot)
    job = _job_document(_read_json(localdirectory / JOB_FILE))
    return _copy_job(localdirectory, sharedroot / "jobs" / job["job_id"], job)


def publish_result(jobdir, job, archivebytes, metrics):
    jobdir = _safe_path(jobdir)
    job = _match_job(jobdir, job)
    if not isinstance(archivebytes, bytes) or not archivebytes or not isinstance(metrics, dict):
        raise ValueError("Historical result requires nonempty archive bytes and a metric dictionary")
    record = {
        "format": FORMAT,
        "job_id": job["job_id"],
        "run_uuid": job["metadata"]["run_uuid"],
        "config_sha256": job["metadata"]["config_sha256"],
        "archive_sha256": hashlib.sha256(archivebytes).hexdigest(),
        "archive_bytes": len(archivebytes),
        "metrics": json.loads(canonical(metrics)),
    }
    if _safe_path(jobdir / "failure.json").exists():
        raise RuntimeError("Historical job already has a failure receipt")
    marker, archive = _safe_path(jobdir / RESULT_FILE), _safe_path(jobdir / ARCHIVE_FILE)
    if marker.exists():
        if _read_json(marker) != record or load_result(jobdir, job) != archivebytes:
            raise FileExistsError("Historical job already has a different result")
        return record
    if archive.exists() and archive.read_bytes() != archivebytes:
        raise FileExistsError("An incomplete historical result contains conflicting archive data")
    if not archive.exists():
        _atomic_write(archive, archivebytes)
    _atomic_write(marker, canonical(record))
    return record


def load_result(jobdir, job):
    jobdir = _safe_path(jobdir)
    job = _match_job(jobdir, job)
    failure = _safe_path(jobdir / "failure.json")
    if failure.exists():
        record = _read_json(failure)
        if not isinstance(record, dict) or record.get("job_id", job["job_id"]) != job["job_id"]:
            raise ValueError("Historical failure receipt belongs to a different job")
        raise RuntimeError(f"Historical rollout worker failed: {canonical(record).decode()}")
    marker = jobdir / RESULT_FILE
    if not marker.exists():
        return None
    record = _read_json(marker)
    if not isinstance(record, dict) or set(record) != {
        "format",
        "job_id",
        "run_uuid",
        "config_sha256",
        "archive_sha256",
        "archive_bytes",
        "metrics",
    }:
        raise ValueError("Invalid historical result receipt")
    for key, expected in (
        ("format", FORMAT),
        ("job_id", job["job_id"]),
        ("run_uuid", job["metadata"]["run_uuid"]),
        ("config_sha256", job["metadata"]["config_sha256"]),
    ):
        if record[key] != expected:
            raise ValueError("Historical result belongs to a different run, configuration, or job")
    if (
        type(record["archive_bytes"]) is not int
        or record["archive_bytes"] < 1
        or not isinstance(record["metrics"], dict)
    ):
        raise ValueError("Invalid historical result payload metadata")
    canonical(record)
    archive = _safe_path(jobdir / ARCHIVE_FILE).read_bytes()
    if len(archive) != record["archive_bytes"] or hashlib.sha256(archive).hexdigest() != record["archive_sha256"]:
        raise ValueError("Historical rollout archive checksum mismatch")
    return archive


def pin_job(localexport, checkpointdestination):
    localexport = _safe_path(localexport)
    job = _job_document(_read_json(localexport / JOB_FILE))
    return _copy_job(localexport, checkpointdestination, job, pin=True)
