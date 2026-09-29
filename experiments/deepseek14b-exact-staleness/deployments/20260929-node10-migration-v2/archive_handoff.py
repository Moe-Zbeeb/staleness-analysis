import json
import os
import select
import signal
import time
from pathlib import Path

assert os.environ["SLURM_JOB_ID"] == "2145806"
pid = 2377637
proc = Path("/proc") / str(pid)
assert proc.stat().st_uid == os.getuid()
assert (proc / "cmdline").read_bytes().split(b"\0")[:-1] == [b"/usr/bin/python3", b"-u", b"-"]
assert b"SLURM_JOB_ID=2145806" in (proc / "environ").read_bytes().split(b"\0")
handle = os.pidfd_open(pid)
archive = Path("/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/archives/node10-model-caches-20260929")
source = Path("/tmp/prime-rl-2142210/Qwen3-14B-40c06982")
retired = source.with_name(source.name + ".verified-archive-20260929")
receipt = archive / "prime-rl-2142210/verified.json"
deadline = time.monotonic() + 3600
while not (receipt.exists() and not source.exists() and not retired.exists()):
    if select.select([handle], [], [], 0)[0]:
        raise RuntimeError("Archive worker exited before the verified first copy")
    if time.monotonic() >= deadline:
        raise TimeoutError("First archive did not finish; worker unchanged")
    time.sleep(0.2)
verified = json.loads(receipt.read_text())
assert verified["source"] == str(source)
signal.pidfd_send_signal(handle, signal.SIGKILL)
exited = bool(select.select([handle], [], [], 30)[0])
result = {"status":"old_archive_worker_killed_after_verified_first_copy","pid":pid,"exited":exited,"receipt":str(receipt),"time":time.time()}
control = Path("/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek15b-k256-node10-migration-20260929-v2")
(control / "archive-handoff.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
os.close(handle)
