import json
import math
import os
import sys
from pathlib import Path


def warn(message):
    print(f"[study-metrics] {message}", file=sys.stderr, flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def scalar(value):
    if not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def valid_step(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class JsonlTail:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.identity = None

    def read(self):
        try:
            stream = self.path.open("rb")
        except FileNotFoundError:
            return []
        with stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            if self.identity is not None and (identity != self.identity or stat.st_size < self.offset):
                raise ValueError(f"Append-only metric file changed: {self.path.name}")
            self.identity = identity
            stream.seek(self.offset)
            data = stream.read(4 * 1024 * 1024)
        end = data.rfind(b"\n")
        if end < 0:
            if len(data) == 4 * 1024 * 1024:
                raise ValueError(f"Oversized metric record: {self.path.name}")
            return []
        self.offset += end + 1
        records = []
        for line in data[: end + 1].splitlines():
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError):
                warn(f"Skipped malformed record in {self.path.name}")
                continue
            if isinstance(record, dict):
                records.append(record)
        return records


def terminal_status(output, allow_completion=True):
    path = output / "run-status.json"
    if path.is_file():
        status = read_json(path).get("status")
        if status not in {"finished", "crashed", "killed"}:
            raise ValueError("Invalid launcher termination status")
        return status
    if allow_completion and (output / "study-complete.json").is_file():
        return "finished"
    return None
