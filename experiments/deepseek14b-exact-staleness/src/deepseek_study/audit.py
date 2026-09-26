import json
from pathlib import Path


def audit(directory):
    directory = Path(directory)
    protocol = json.loads((directory / "configs" / "protocol.json").read_text())
    lag = protocol["lag"]
    run = json.loads((directory / "run.json").read_text())
    rows = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
    if not rows:
        raise ValueError("No completed optimizer updates have been recorded")
    ids, previous, warmup, steady = set(), run["starting_step"], 0, 0
    for row in rows:
        step = row["step"]
        if step != previous + 1 or step > protocol["max_steps"]:
            raise ValueError("Update log contains duplicate or missing steps")
        expected = 0 if step <= lag else lag
        if row["learner_version"] != step - 1 or row["learner_version"] - row["behavior_version"] != expected:
            raise ValueError("Training provenance disagrees with the required optimizer-update age")
        if row["age_min"] != expected or row["age_max"] != expected:
            raise ValueError("A training update contains the wrong rollout age")
        if row["warmup"] != (step <= lag) or row["responses"] != protocol["responses_per_update"]:
            raise ValueError("Update phase or cohort size differs from the run protocol")
        current = set(row["response_ids"])
        if len(current) != row["responses"] or current & ids:
            raise ValueError("A response was duplicated or trained on more than once")
        ids.update(current)
        warmup += step <= lag
        steady += step > lag
        previous = step
    complete_path = directory / "study-complete.json"
    complete = complete_path.is_file()
    if complete:
        marker = json.loads(complete_path.read_text())
        if (
            marker["step"] != previous
            or previous != protocol["max_steps"]
            or marker["lag"] != lag
            or marker["identity_sha256"] != run["identity_sha256"]
        ):
            raise ValueError("Completion marker does not match the audited run")
    return {
        "updates_in_this_run": len(rows),
        "last_completed_step": previous,
        "warmup_updates": warmup,
        "requested_lag": lag,
        "exact_staleness_updates": steady,
        "unique_trained_responses": len(ids),
        "complete": complete,
    }
