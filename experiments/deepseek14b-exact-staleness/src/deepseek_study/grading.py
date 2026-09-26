import asyncio
import json
import sys
import time
import uuid
import weakref


class GraderFailure(RuntimeError):
    pass


class Worker:
    def __init__(self, command=None):
        self.command = command or [sys.executable, "-m", "deepseek_study.grader_worker"]
        self.process = None

    async def close(self):
        if self.process is None:
            return
        if self.process.returncode is None:
            try:
                self.process.kill()
            except ProcessLookupError:
                pass
        await self.process.wait()
        self.process = None

    async def call(self, request, timeout):
        try:
            if self.process is None:
                self.process = await asyncio.create_subprocess_exec(
                    *self.command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, limit=4 * 1024 * 1024
                )
                ready = await asyncio.wait_for(self.process.stdout.readline(), 60)
                if json.loads(ready) != {"ready": True}:
                    raise GraderFailure("Invalid grader startup acknowledgement")
            async with asyncio.timeout(timeout):
                self.process.stdin.write((json.dumps(request) + "\n").encode())
                await self.process.stdin.drain()
                line = await self.process.stdout.readline()
                if not line:
                    raise GraderFailure("Grader worker exited without a result")
                response = json.loads(line)
                if response.get("id") != request["id"]:
                    raise GraderFailure("Grader response identity mismatch")
                if response.get("status") not in {"ok", "error", "reference_rejected"}:
                    raise GraderFailure("Invalid grader response status")
                return response
        except BaseException:
            await self.close()
            raise


class GraderPool:
    def __init__(self, workers=4, timeout=10, retries=1, command=None):
        self.workers = [Worker(command) for _ in range(workers)]
        self.available = asyncio.Queue()
        for worker in self.workers:
            self.available.put_nowait(worker)
        self.timeout = timeout
        self.retries = retries
        self.closed = False

    async def call(self, request):
        if self.closed:
            raise GraderFailure("Grader pool is closed")
        request = {**request, "id": uuid.uuid4().hex}
        worker = await self.available.get()
        errors = []
        started = time.monotonic()
        try:
            for attempt in range(self.retries + 1):
                try:
                    response = await worker.call(request, self.timeout)
                    if response["status"] == "error":
                        raise GraderFailure(response["reason"])
                    response.update(attempts=attempt + 1, errors=errors, seconds=time.monotonic() - started)
                    return response
                except (Exception,) as error:
                    errors.append({"type": type(error).__name__, "reason": str(error)[:256]})
                    await worker.close()
            raise GraderFailure(json.dumps({"message": "Persistent grader failure on saved input", "errors": errors}))
        finally:
            self.available.put_nowait(worker)

    async def close(self):
        self.closed = True
        await asyncio.gather(*(worker.close() for worker in self.workers))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


_POOLS = weakref.WeakKeyDictionary()


def get_pool(workers, timeout, retries):
    loop = asyncio.get_running_loop()
    key = (workers, timeout, retries)
    pools = _POOLS.setdefault(loop, {})
    if key not in pools:
        pools[key] = GraderPool(workers, timeout, retries)
    return pools[key]
