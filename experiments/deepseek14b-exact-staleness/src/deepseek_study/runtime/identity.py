import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

from deepseek_study import PRIME_COMMIT
from deepseek_study.runtime.checkpoints import atomic_write
from deepseek_study.dataset.prepare import digest, load_manifest


def source_files(root):
    root = Path(root)
    paths = [root / "pyproject.toml", *sorted((root / "src").rglob("*.py")), *sorted((root / "scripts").glob("*.py"))]
    if (root / "manifests/model.json").is_file():
        paths.append(root / "manifests/model.json")
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def capture(root, study):
    root = Path(root)
    packages = (
        "prime-rl",
        "torch",
        "transformers",
        "vllm",
        "math-verify",
        "latex2sympy2-extended",
        "sympy",
        "msgspec",
        "pyarrow",
        "runboard",
    )
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    body = {
        "format": 1,
        "source_files": source_files(root),
        "prime_commit": PRIME_COMMIT,
        "lock_sha256": hashlib.sha256((root / "vendor" / "prime-rl" / "uv.lock").read_bytes()).hexdigest(),
        "runtime": versions,
        "python": platform.python_version(),
        "data_sha256": load_manifest(study.data_manifest)["sha256"],
    }
    return {**body, "sha256": digest(body)}


def snapshot(root, destination, identity):
    root, destination = Path(root), Path(destination)
    if source_files(root) != identity["source_files"]:
        raise ValueError("Study source changed while preparing the run")
    destination.mkdir(parents=True, exist_ok=False)
    for name, expected in identity["source_files"].items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Source snapshot changed during copying")
    shutil.copyfile(root / "vendor" / "prime-rl" / "uv.lock", destination / "prime-rl.uv.lock")
    atomic_write(destination / "identity.json", (json.dumps(identity, indent=2) + "\n").encode())


def read_identity(path):
    value = json.loads(Path(path).read_text())
    if value.get("format") != 1 or value.get("sha256") != digest(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise ValueError("Run identity failed integrity validation")
    return value
