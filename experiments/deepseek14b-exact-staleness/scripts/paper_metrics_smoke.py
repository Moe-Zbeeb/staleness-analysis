import json
import os
import socket
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import torch
from runboard.config import read_server_info

from deepseek_study.recipe import baseline
from deepseek_study.tracking.observer import observe_papers
from deepseek_study.tracking.runboard import observe
from deepseek_study.tracking.tokens import PaperTokenExporter


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def main():
    root = Path(__file__).resolve().parents[1]
    name = "paper-metrics-smoke-" + os.environ.get("SLURM_JOB_ID", uuid.uuid4().hex[:12])
    output = root / "diagnostics/cluster" / name
    study = baseline(32, root=root).model_copy(
        update={"output_dir": output, "prompts_per_update": 2, "responses_per_prompt": 4, "trainer_gpus": 2}
    )
    output.mkdir(parents=True, exist_ok=False)
    write(output / "configs/study.json", study.model_dump(mode="json"))
    write(
        output / "run.json",
        {
            "run_uuid": uuid.uuid4().hex,
            "starting_step": 0,
            "resume_from": None,
            "identity_sha256": "synthetic-diagnostic",
            "config_sha256": study.fingerprint(),
            "diagnostic_only": True,
        },
    )
    for rank, response_count in ((0, 3), (1, 5)):
        advantages = torch.tensor([0.0, 1.0, 1.0] * response_count)
        if rank:
            advantages = -advantages
        advantages[1:3] = 0
        behavior = torch.full((3 * response_count,), -2.0)
        current = behavior + (0.4 if rank == 0 else -0.4)
        batch = {
            "input_ids": torch.arange(3 * response_count).reshape(1, -1),
            "position_ids": torch.tensor([0, 1, 2] * response_count).reshape(1, -1),
            "loss_mask": torch.tensor([False, True, True] * response_count).reshape(1, -1),
            "inference_logprobs": behavior,
            "advantages": advantages,
        }
        exporter = PaperTokenExporter(output, rank, study.clip_epsilon)
        exporter.export(
            33,
            0,
            batch,
            {"logprobs": current, "entropy": torch.full_like(current, 0.3 + rank)},
            [3] * response_count,
            None,
        )
        exporter.mark_stable()
    write(
        output / "updates.jsonl",
        {
            "step": 33,
            "learner_version": 32,
            "behavior_version": 0,
            "warmup": False,
            "age_min": 32,
            "age_max": 32,
            "responses": 8,
            "mean_reward": 0.375,
        },
    )
    write(output / "metrics.jsonl", {"producer": "trainer", "step": 33, "optim/grad_norm": 0.5})
    write(output / "run-status.json", {"status": "finished"})
    archive = observe_papers(output, once=True)
    os.environ["RUNBOARD_PROJECT"] = "staleness-analysis-checks"
    info = observe(output, once=True)
    if info["mode"] != "http":
        raise RuntimeError("Live Runboard verification requires the configured HTTP backend")
    settings = read_server_info()
    server = os.environ.get("RUNBOARD_SERVER") or settings.get("url")
    token = os.environ.get("RUNBOARD_TOKEN") or settings.get("token")
    query = urllib.parse.urlencode({"project": info["project"], "run": info["run_id"]})
    request = urllib.request.Request(
        server.rstrip("/") + "/api/metrics?" + query,
        headers={"Authorization": "Bearer " + token, "User-Agent": "runboard/0.2.0"},
    )
    for attempt in range(10):
        with urllib.request.urlopen(request, timeout=10) as response:
            rows = json.load(response)["rows"]
        if len(rows) == 3:
            break
        time.sleep(1)
    expected = json.loads((output / "paper-metrics.jsonl").read_text())["metrics"]
    assert expected["clip/fraction"] == 0.75
    assert expected["gradient_signal/noncontributing_token_fraction"] == 1
    assert expected["gradient_signal/zero_advantage_fraction"] == 0.25
    paper_rows = [row for row in rows if "paper/mismatch/m2" in row]
    if len(rows) != 3 or len(paper_rows) != 1 or paper_rows[0]["_step"] != 33:
        raise RuntimeError("Hosted Runboard row/step verification failed")
    if any(paper_rows[0].get("paper/" + key) != value for key, value in expected.items()):
        raise RuntimeError("Hosted Runboard scalar values differ from archived evidence")
    result = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": socket.gethostname(),
        "output": str(output),
        "archive": archive,
        "runboard": info,
        "verified_rows": len(rows),
        "verified_paper_scalars": len(expected),
        "training_started": False,
        "gpu_used": False,
    }
    write(root / "diagnostics/cluster/paper-metrics-validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
