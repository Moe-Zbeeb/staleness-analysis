import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .common import digest, immutable_json, lock, read, write


class LostLease(RuntimeError):
    pass


class Queue:
    """Small control ledger with NFS advisory locking; large outputs live separately.

    Expired attempts are fenced by a new UUID. Heartbeats cannot resurrect a lease.
    A task may execute twice after a crash, but only its current owner can publish.
    """

    def __init__(self, root, clock=time.time):
        self.root = Path(root)
        self.clock = clock
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def transaction(self):
        with lock(self.root / "queue.lock"):
            path = self.root / "queue.json"
            state = read(path) if path.exists() else {"format": 1, "tasks": {}, "served": {}, "counter": 0}
            if state["format"] != 1:
                raise ValueError("Unknown queue format")
            yield state
            write(path, state)

    def enqueue(self, spec, depends_on=None, max_attempts=3):
        task_id = digest(spec)
        with self.transaction() as state:
            old = state["tasks"].get(task_id)
            if old:
                if old["spec"] != spec or old["depends_on"] != depends_on:
                    raise ValueError("Task identity conflict")
                return task_id
            state["tasks"][task_id] = {
                "spec": spec, "depends_on": depends_on, "state": "queued", "attempts": 0,
                "max_attempts": max_attempts, "created": self.clock(), "errors": [],
            }
        return task_id

    def _expire(self, state):
        for task in state["tasks"].values():
            if task["state"] == "running" and task["expires"] <= self.clock():
                task["errors"].append({"time": self.clock(), "error": "lease_expired"})
                task["state"] = "failed" if task["attempts"] >= task["max_attempts"] else "queued"
            dependency = task["depends_on"]
            if task["state"] == "queued" and dependency and state["tasks"][dependency]["state"] == "failed":
                task["blocked_by"] = dependency
            else:
                task.pop("blocked_by", None)

    def claim(self, kind, owner, seconds=180):
        with self.transaction() as state:
            self._expire(state)
            candidates = []
            for key, task in state["tasks"].items():
                if task["state"] != "queued" or task["spec"]["kind"] != kind:
                    continue
                dep = task["depends_on"]
                if dep and state["tasks"][dep]["state"] != "done":
                    continue
                spec = task["spec"]
                served = state["served"].get(f"{kind}:{spec['run_id']}", 0)
                candidates.append(((served, spec["step"], task["created"], key), key, task))
            if not candidates:
                return None
            _, key, task = min(candidates, key=lambda item: item[0])
            task.update(state="running", attempts=task["attempts"] + 1, token=uuid.uuid4().hex,
                        owner=owner, expires=self.clock() + seconds, last_progress=self.clock())
            state["counter"] += 1
            state["served"][f"{kind}:{task['spec']['run_id']}"] = state["counter"]
            return {"id": key, "token": task["token"], "spec": task["spec"], "attempt": task["attempts"]}

    def _owned(self, state, claim):
        task = state["tasks"][claim["id"]]
        if task["state"] != "running" or task.get("token") != claim["token"] or task["expires"] <= self.clock():
            raise LostLease(claim["id"])
        return task

    def heartbeat(self, claim, seconds=180):
        with self.transaction() as state:
            task = self._owned(state, claim)
            task["expires"] = self.clock() + seconds
            return task["last_progress"]

    def progress(self, claim):
        with self.transaction() as state:
            self._owned(state, claim)["last_progress"] = self.clock()

    def finish(self, claim, receipt):
        with self.transaction() as state:
            task = self._owned(state, claim)
            immutable_json(self.root / "results" / claim["id"] / "complete.json", receipt)
            task.update(state="done", finished=self.clock(), receipt_sha256=digest(receipt))

    def fail(self, claim, error):
        with self.transaction() as state:
            task = self._owned(state, claim)
            task["errors"].append({"time": self.clock(), "error": str(error)[:2000]})
            task["state"] = "failed" if task["attempts"] >= task["max_attempts"] else "queued"

    def save_sample(self, claim, sample_id, stage, value):
        if len(sample_id) != 64 or any(c not in "0123456789abcdef" for c in sample_id):
            raise ValueError("Invalid sample ID")
        if stage not in {"raw", "grade"}:
            raise ValueError("Invalid sample stage")
        with self.transaction() as state:
            task = self._owned(state, claim)
            immutable_json(self.sample_path(claim["id"], sample_id, stage), {"data": value, "sha256": digest(value)})
            task["last_progress"] = self.clock()

    def sample_path(self, task_id, sample_id, stage):
        return self.root / "results" / task_id / stage / f"{sample_id}.json"

    def read_sample(self, task_id, sample_id, stage):
        record = read(self.sample_path(task_id, sample_id, stage))
        if digest(record["data"]) != record["sha256"]:
            raise ValueError("Saved sample checksum mismatch")
        return record["data"]

    def snapshot(self):
        with self.transaction() as state:
            self._expire(state)
            return state

    def retry(self, task_id):
        with self.transaction() as state:
            task = state["tasks"][task_id]
            if task["state"] != "failed":
                raise ValueError("Only failed tasks can be explicitly retried")
            task.update(state="queued", attempts=0)
