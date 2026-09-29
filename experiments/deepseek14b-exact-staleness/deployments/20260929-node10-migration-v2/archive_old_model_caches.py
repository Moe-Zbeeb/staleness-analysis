import hashlib
import json
import os
import shutil
import stat
import time
from pathlib import Path

root = Path("/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/archives/node10-model-caches-20260929")
expected = {"2142210": "Qwen3-14B-40c06982", "2142188": "Qwen3-14B-40c06982", "2142052": "Qwen2.5-Math-7B-b101308f", "2142033": "Qwen2.5-Math-7B-b101308f", "2142029": "Qwen2.5-Math-7B-b101308f", "2142032": "Qwen2.5-3B-3aab1f19", "2142027": "Qwen2.5-3B-3aab1f19"}
assert os.environ.get("SLURMD_NODENAME") == "deep-chungus-10"
assert os.getuid() == 29562
assert not root.exists()
root.mkdir(parents=True)
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()
def inventory(path):
    result = {}
    for item in sorted(path.rglob("*")):
        st = item.lstat()
        if st.st_uid != os.getuid() or item.is_symlink() or not (item.is_file() or item.is_dir()):
            raise ValueError(str(item))
        if item.is_file():
            result[str(item.relative_to(path))] = {"size": st.st_size, "sha256": digest(item)}
    return result
for job, model in expected.items():
    source = Path("/tmp") / ("prime-rl-" + job) / model
    target = root / ("prime-rl-" + job) / model
    assert source.is_dir() and not source.is_symlink() and source.stat().st_uid == os.getuid()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            command = (proc / "cmdline").read_bytes()
            if str(source).encode() in command or str(source.parent).encode() in command:
                raise RuntimeError("Old cache referenced by running process " + proc.name)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    before = inventory(source)
    assert before and any(name.endswith(".safetensors") for name in before)
    print(json.dumps({"phase":"copying","source":str(source),"bytes":sum(x["size"] for x in before.values())}), flush=True)
    shutil.copytree(source, target)
    after = inventory(target)
    if before != after or inventory(source) != before:
        raise RuntimeError("Archive verification failed")
    receipt = {"source":str(source),"archive":str(target),"files":before,"verified_at":time.time()}
    with (target.parent / "verified.json").open("x") as f:
        json.dump(receipt,f,indent=2)
        f.flush()
        os.fsync(f.fileno())
    retired = source.with_name(source.name + ".verified-archive-20260929")
    assert not retired.exists()
    source.rename(retired)
    shutil.rmtree(retired)
    print(json.dumps({"phase":"archived_and_reclaimed","source":str(source),"free":shutil.disk_usage("/tmp").free}), flush=True)
