import asyncio
import sys

import pytest

from deepseek_study.dataset.grading import GraderFailure, GraderPool


async def test_actual_isolated_worker_grades_and_preserves_failure_reason():
    async with GraderPool(workers=2) as pool:
        requests = [
            {
                "operation": "grade",
                "arguments": {
                    "raw_completion": response,
                    "answer": "4",
                    "truncated": False,
                    "truncated_reward": "zero",
                    "timeout": 5,
                },
            }
            for response in ("</think>\\boxed{4}", "still thinking")
        ]
        correct, missing = await asyncio.gather(*(pool.call(request) for request in requests))
        assert correct["result"]["reward"] == 1
        assert missing["result"]["reward"] == 0
        assert missing["result"]["reason"] == "missing_reasoning_close"
        assert correct["attempts"] == 1


async def test_worker_deadline_kills_process_and_fails_after_bounded_retries():
    script = (
        "import json,sys,time\nprint(json.dumps({'ready':True}),flush=True)\nfor line in sys.stdin:\n time.sleep(10)\n"
    )
    async with GraderPool(workers=1, timeout=0.05, retries=1, command=[sys.executable, "-u", "-c", script]) as pool:
        with pytest.raises(GraderFailure, match="Persistent grader failure"):
            await pool.call({"operation": "reference", "answer": "4", "timeout": 1})
        assert pool.workers[0].process is None


async def test_retry_reuses_identical_input_and_records_attempts(tmp_path):
    marker = tmp_path / "input.json"
    script = "import json,sys,pathlib\np=pathlib.Path(sys.argv[1])\nprint(json.dumps({'ready':True}),flush=True)\nfor line in sys.stdin:\n r=json.loads(line)\n if not p.exists():\n  p.write_text(line)\n  sys.exit(1)\n assert json.loads(p.read_text())==r\n print(json.dumps({'id':r['id'],'status':'ok','result':{'reward':1}}),flush=True)\n"
    async with GraderPool(
        workers=1, timeout=2, retries=1, command=[sys.executable, "-u", "-c", script, str(marker)]
    ) as pool:
        response = await pool.call({"operation": "grade", "arguments": {"saved_response": "unchanged"}})
        assert response["attempts"] == 2
        assert len(response["errors"]) == 1


async def test_cancellation_terminates_the_inflight_worker():
    script = (
        "import json,sys,time\nprint(json.dumps({'ready':True}),flush=True)\nfor line in sys.stdin:\n time.sleep(10)\n"
    )
    async with GraderPool(workers=1, timeout=20, command=[sys.executable, "-u", "-c", script]) as pool:
        task = asyncio.create_task(pool.call({"operation": "reference"}))
        while pool.workers[0].process is None:
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert pool.workers[0].process is None
