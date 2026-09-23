import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    destination = Path(spec["models_dir"]) / spec["base_model_directory"]
    record = destination / "MODEL_INFO.json"
    expected = {"repository": spec["model"], "revision": spec["model_revision"]}
    if record.exists():
        previous = json.loads(record.read_text())
        if any(previous[key] != value for key, value in expected.items()):
            raise RuntimeError("Existing model provenance does not match the requested pin")
    info = HfApi(token=False).model_info(spec["model"], revision=spec["model_revision"])
    if info.sha != spec["model_revision"]:
        raise RuntimeError("Model repository did not resolve to the requested revision")
    snapshot_download(
        repo_id=spec["model"],
        revision=spec["model_revision"],
        local_dir=destination,
        token=False,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE", "README.md"],
        max_workers=4,
    )
    files = {}
    for path in sorted(destination.iterdir()):
        if path.is_file() and path.name != record.name:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files[path.name] = {"bytes": path.stat().st_size, "sha256": digest}
    if "model.safetensors" not in files:
        raise RuntimeError("The pinned 1.5B model weights were not downloaded")
    metadata = {**expected, "path": str(destination), "files": files}
    temporary = record.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(record)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
