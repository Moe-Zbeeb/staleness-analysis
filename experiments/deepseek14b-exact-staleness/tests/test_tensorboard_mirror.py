import json
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from deepseek_study.tracking.archive import TensorBoardMirror, sha256
from deepseek_study.tracking import archive


def test_tensorboard_events_mirror_incrementally_and_load(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "run.json").write_text(json.dumps({"run_uuid": "tensorboard-test"}))
    mirror = TensorBoardMirror(output, tmp_path / "xfs")
    with SummaryWriter(str(output / "tensorboard/training")) as writer:
        writer.add_scalar("loss", 0.75, 1)
        writer.flush()
        mirror.poll()
        first_size = next((mirror.destination / "tensorboard").rglob("events.*")).stat().st_size
        writer.add_scalar("loss", 0.25, 2)
        writer.flush()
        mirror.poll()
        assert next((mirror.destination / "tensorboard").rglob("events.*")).stat().st_size > first_size
    result = mirror.finish()
    assert result["verified_files"] == 1
    events = EventAccumulator(str(mirror.destination / "tensorboard/training")).Reload()
    assert [(item.step, item.value) for item in events.Scalars("loss")] == [(1, 0.75), (2, 0.25)]
    manifest = json.loads((mirror.destination / "tensorboard-mirror-manifest.json").read_text())["verified_files"]
    for name, record in manifest.items():
        assert sha256(output / name) == record["sha256"]


def test_tensorboard_mirror_rejects_modified_history(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "run.json").write_text(json.dumps({"run_uuid": "tensorboard-test"}))
    with SummaryWriter(str(output / "tensorboard/training")) as writer:
        writer.add_scalar("loss", 0.75, 1)
    mirror = TensorBoardMirror(output, tmp_path / "xfs")
    mirror.poll()
    target = next((mirror.destination / "tensorboard").rglob("events.*"))
    data = target.read_bytes()
    target.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(ValueError, match="prefix differs"):
        TensorBoardMirror(output, tmp_path / "xfs").poll()


def test_two_observers_can_reserve_existing_mirror_concurrently(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    (output / "run.json").write_text(json.dumps({"run_uuid": "shared"}))
    root = tmp_path / "xfs"
    expected = archive.reserve_mirror(output, root)
    original = archive.atomic_write
    barrier = threading.Barrier(2)

    def write(path, data):
        original(path, data)
        if path.name.startswith(".write-probe"):
            barrier.wait(timeout=5)

    monkeypatch.setattr(archive, "atomic_write", write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(archive.reserve_mirror, output, root) for _ in range(2)]
        assert [future.result(timeout=10) for future in futures] == [expected, expected]
    assert not list(expected.glob(".write-probe*"))
