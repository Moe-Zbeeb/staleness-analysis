import json
import os
import signal
import time
from pathlib import Path

control = Path("/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek15b-k256-node10-migration-20260929-v2")
pid = 2359994
process = Path("/proc") / str(pid)
assert os.environ["SLURM_JOB_ID"] == "2145806"
assert process.stat().st_uid == os.getuid()
assert (process / "cmdline").read_bytes().split(b"\0")[:-1] == [b"/usr/bin/python3", str(control / "prefetch.py").encode()]
assert b"SLURM_JOB_ID=2145806" in (process / "environ").read_bytes().split(b"\0")
handle = os.pidfd_open(pid)
signal.pidfd_send_signal(handle, signal.SIGSTOP)
print(json.dumps({"status":"prefetch_paused_for_storage","pid":pid}),flush=True)
try:
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        if (control / "controls-finalized.json").exists():
            archive = Path("/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/archives/node10-model-caches-20260929")
            if len(list(archive.glob("*/verified.json"))) == 7:
                break
        time.sleep(5)
finally:
    signal.pidfd_send_signal(handle, signal.SIGCONT)
    os.close(handle)
    print(json.dumps({"status":"prefetch_resumed","pid":pid}),flush=True)
