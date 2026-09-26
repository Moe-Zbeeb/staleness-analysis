import hashlib
import json
import os
import shutil
from pathlib import Path

from deepseek_study.runtime.checkpoints import atomic_write


JOURNALS = (
    "metrics.jsonl",
    "updates.jsonl",
    "generations.jsonl",
    "grading.jsonl",
    "paper-metrics.jsonl",
    "evaluation-metrics.jsonl",
)
FILES = ("run.json", "run-status.json", "study-complete.json", "preflight.json", "source/identity.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def reserve_mirror(output, root):
    if root is None:
        return None
    output = Path(output).resolve()
    destination = Path(root).resolve() / output.name
    owner = json.loads((output / "run.json").read_text())
    identity = {"source": str(output), "run_uuid": owner["run_uuid"]}
    marker = destination / "mirror-owner.json"
    if destination.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise FileExistsError("Metric mirror belongs to another run; choose a unique output directory name")
    else:
        destination.mkdir(parents=True, exist_ok=False)
        atomic_write(marker, json.dumps(identity).encode())
    probe = destination / ".write-probe"
    atomic_write(probe, b"metric archive preflight\n")
    probe.unlink()
    return destination


class MetricMirror:
    def __init__(self, output, root):
        self.output = Path(output)
        self.destination = reserve_mirror(output, root)
        self.copied = set()
        self.checked_journals = set()
        self.verified = {}

    def poll(self):
        if self.destination is None:
            return
        files = [self.output / name for name in FILES]
        for folder in ("configs", "paper", "evaluations"):
            files.extend((self.output / folder).rglob("*.json"))
            files.extend((self.output / folder).rglob("*.npz"))
            files.extend((self.output / folder).rglob("*.jsonl"))
        for source in files:
            if not source.is_file() or source in self.copied:
                continue
            target = self.destination / source.relative_to(self.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            with source.open("rb") as incoming, temporary.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 4 * 1024 * 1024)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            os.replace(temporary, target)
            digest = sha256(target)
            if digest != sha256(source):
                raise ValueError(f"Metric mirror checksum mismatch: {source.name}")
            self.verified[str(source.relative_to(self.output))] = {"bytes": target.stat().st_size, "sha256": digest}
            self.copied.add(source)
        for name in JOURNALS:
            source, target = self.output / name, self.destination / name
            if not source.is_file():
                continue
            offset = target.stat().st_size if target.exists() else 0
            if source.stat().st_size < offset:
                raise ValueError("Metric mirror is ahead of its append-only source")
            with source.open("rb") as incoming:
                if name not in self.checked_journals and offset:
                    remaining = offset
                    with target.open("rb") as previous:
                        while remaining:
                            count = min(remaining, 4 * 1024 * 1024)
                            if incoming.read(count) != previous.read(count):
                                raise ValueError("Metric mirror prefix differs from its source")
                            remaining -= count
                self.checked_journals.add(name)
                incoming.seek(offset)
                with target.open("ab") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 4 * 1024 * 1024)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())

    def finish(self):
        if self.destination is None:
            return None
        self.poll()
        manifest = dict(self.verified)
        for name in JOURNALS:
            target = self.destination / name
            if not target.is_file():
                continue
            digest = sha256(target)
            if digest != sha256(self.output / name):
                raise ValueError(f"Metric mirror checksum mismatch: {name}")
            manifest[name] = {"bytes": target.stat().st_size, "sha256": digest}
        atomic_write(
            self.destination / "mirror-manifest.json",
            json.dumps({"schema_version": 1, "verified_files": manifest}, indent=2).encode(),
        )
        return {"directory": str(self.destination), "verified_files": len(manifest)}
