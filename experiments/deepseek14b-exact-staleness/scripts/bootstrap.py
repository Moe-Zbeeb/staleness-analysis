import argparse
import os
import subprocess
import tomllib
from pathlib import Path

PIN = "ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1"


def run(*args, cwd=None, env=None):
    subprocess.run(args, cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--attention", choices=["fa2", "fa3", "fa4"], default="fa2")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    vendor = root / "vendor" / "prime-rl"
    git_environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    if not vendor.exists():
        vendor.parent.mkdir(exist_ok=True)
        run(
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "https://github.com/PrimeIntellect-ai/prime-rl.git",
            str(vendor),
            env=git_environment,
        )
        run("git", "checkout", "--detach", PIN, cwd=vendor, env=git_environment)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vendor, text=True).strip()
    if actual != PIN or subprocess.check_output(["git", "diff", "HEAD"], cwd=vendor):
        raise RuntimeError("Existing official checkout differs from the pinned source; refusing to overwrite it")
    run(
        "git",
        "-c",
        "url.https://github.com/.insteadOf=git@github.com:",
        "submodule",
        "update",
        "--init",
        "--",
        "deps/verifiers",
        "deps/renderers",
        "deps/pydantic-config",
        "deps/prime-envs",
        cwd=vendor,
        env=git_environment,
    )
    command = ["uv", "sync", "--frozen", "--python", "3.12"]
    if args.gpu:
        if os.uname().sysname != "Linux":
            raise RuntimeError("GPU runtime requires Linux")
        command += [
            "--extra",
            "gpu",
            "--extra",
            {"fa2": "flash-attn", "fa3": "flash-attn-3", "fa4": "flash-attn-cute"}[args.attention],
        ]
    run(*command, cwd=vendor)
    python = str(vendor / ".venv" / "bin" / "python")
    if not args.gpu:
        run("uv", "pip", "install", "--python", python, "torch==2.11.0", "prometheus-client==0.25.0")
    dependencies = tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
    runboard = next(requirement for requirement in dependencies if requirement.startswith("runboard @ "))
    run("uv", "pip", "install", "--python", python, "--no-deps", runboard)
    run("uv", "pip", "install", "--python", python, "--no-deps", "-e", str(root))
    print(f"Ready: {vendor / '.venv' / 'bin' / 'deepseek-study'}")


if __name__ == "__main__":
    main()
