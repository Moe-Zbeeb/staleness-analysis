from pathlib import Path

from deepseek_study.config import StudyConfig
from deepseek_study.runtime.checkpoints import verify_components

from .common import digest, file_hash, immutable_json, read, tokenizer_identity
from .protocol import Protocol
from .queue import Queue


def register(root, run, protocol_path):
    root, run = Path(root), Path(run).resolve()
    study = StudyConfig.read(run / "configs/study.json")
    protocol = Protocol.load(protocol_path)
    if study.output_dir.resolve() != run:
        raise ValueError("Study output directory and registration disagree")
    if study.checkpoint_interval % study.checkpoint_keep_interval or study.max_steps % study.checkpoint_keep_interval:
        raise ValueError("Evaluation requires retained checkpoint milestones; current retention may prune pending input")
    if protocol.tokenizer != tokenizer_identity(study.prepared_model_path):
        raise ValueError("Evaluation tokenizer differs from training")
    if protocol.instruction != study.prompt_instruction:
        raise ValueError("Evaluation instruction differs from training")
    source_identity = read(run / "source/identity.json")["sha256"]
    record = {
        "run_id": digest([str(run), study.fingerprint(), source_identity]), "path": str(run),
        "config_sha256": study.fingerprint(), "identity_sha256": source_identity, "lag": study.lag,
        "seed": study.seed, "interval": study.checkpoint_interval, "max_steps": study.max_steps,
        "model": str(study.prepared_model_path.resolve()), "protocol": protocol.identity,
    }
    immutable_json(root / "protocols" / f"{protocol.identity}.json", protocol.model_dump())
    immutable_json(root / "runs" / f"{record['run_id']}-{protocol.identity}.json", record)
    return record


def checkpoint_receipt(path, run=None):
    path = Path(path).resolve()
    marker = read(path / "study/complete.json")
    if marker.get("format") != 2 or path.name != f"step_{marker['step']}":
        raise ValueError("Checkpoint format/step mismatch")
    if run and any(marker[key] != run[key] for key in ("config_sha256", "identity_sha256", "lag")):
        raise ValueError("Checkpoint belongs to another training configuration/source")
    verify_components(path)
    return {"path": str(path), "marker": marker, "marker_sha256": file_hash(path / "study/complete.json")}


def discover(root):
    root, added, errors = Path(root), [], []
    queue = Queue(root)
    for registration in sorted((root / "runs").glob("*.json")):
        run = read(registration)
        for step in range(run["interval"], run["max_steps"] + 1, run["interval"]):
            checkpoint = Path(run["path"]) / "checkpoints" / f"step_{step}"
            if not (checkpoint / "study/complete.json").exists():
                continue
            try:
                source = checkpoint_receipt(checkpoint, run)
                checkpoint_id = digest(source)
                model = str((root / "exports" / checkpoint_id).resolve())
                export_spec = {"kind": "export", "run_id": run["run_id"], "step": step, "source": source,
                               "model_source": run["model"], "model": model, "checkpoint_id": checkpoint_id}
                export_id = queue.enqueue(export_spec)
                eval_spec = {"kind": "evaluate", "run_id": run["run_id"], "step": step, "lag": run["lag"],
                             "training_seed": run["seed"], "model": model, "checkpoint_id": checkpoint_id,
                             "protocol": run["protocol"]}
                protocol = Protocol.load(root / "protocols" / f"{run['protocol']}.json")
                added.extend(enqueue_evaluation(queue, eval_spec, protocol, depends_on=export_id))
            except (OSError, ValueError, KeyError) as error:
                errors.append({"run_id": run["run_id"], "step": step, "error": str(error)})
    return {"tasks": added, "errors": errors}


def enqueue_evaluation(queue, spec, protocol, depends_on=None):
    shards = list(protocol.shards())
    group = digest([spec["run_id"], spec["step"], spec["checkpoint_id"], protocol.identity])
    return [queue.enqueue({**spec, "group": group, "shard": index, "shards_total": len(shards),
                           "question_ids": questions}, depends_on=depends_on)
            for index, questions in enumerate(shards)]
