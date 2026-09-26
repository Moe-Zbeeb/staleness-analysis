import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    selected = [
        root / name
        for name in (
            "pyproject.toml",
            "README.md",
            ".gitignore",
        )
    ]
    for name in ("src", "scripts", "tests", "configs", "manifests", "docs"):
        selected.extend(
            path
            for path in (root / name).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    for name in ("prepared-data-exclusions.json", "prepared-data-summary.json", "cluster-validation.json"):
        selected.append(root / "diagnostics" / name)
    prepared = root / "assets/train-manifest.json"
    if prepared.is_file():
        selected.append(prepared)
    selected = sorted(set(selected))
    checksums = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in selected}
    destination = root / "dist/deepseek-staleness-study.tar.gz"
    destination.parent.mkdir(exist_ok=True)
    with (
        destination.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in selected:
            info = tarfile.TarInfo(str(path.relative_to(root)))
            contents = path.read_bytes()
            info.size = len(contents)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(contents))
        contents = (json.dumps(checksums, indent=2) + "\n").encode()
        info = tarfile.TarInfo("PACKAGE_SHA256.json")
        info.size = len(contents)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(contents))
    checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(f"{checksum}  {destination.name}\n")
    print(json.dumps({"path": str(destination), "bytes": destination.stat().st_size, "sha256": checksum}, indent=2))


if __name__ == "__main__":
    main()
