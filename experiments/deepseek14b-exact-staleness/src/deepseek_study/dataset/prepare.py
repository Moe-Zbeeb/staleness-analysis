import asyncio
import hashlib
import json
from pathlib import Path

from deepseek_study import DATASET_SHA256, MODEL_REVISION
from deepseek_study.dataset.assets import dataset_source, read_rows, validate_grading_format
from deepseek_study.dataset.grading import GraderPool
from deepseek_study.dataset.rewards import reward_identity, tokenizer
from deepseek_study.dataset.sources import source_for_sha256
from deepseek_study.runtime.checkpoints import atomic_write


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def data_contract(
    prompt_instruction,
    prompt_max_tokens,
    reward_timeout_seconds,
    reasoning_required=True,
    source_sha256=DATASET_SHA256,
):
    source = source_for_sha256(source_sha256)
    contract = {
        "source_sha256": source.sha256,
        "model_revision": MODEL_REVISION,
        "prompt_instruction": prompt_instruction,
        "prompt_max_tokens": prompt_max_tokens,
        "reward_timeout_seconds": reward_timeout_seconds,
        "reasoning_required": reasoning_required,
        "reward": reward_identity(),
        "oversized_prompts": "exclude_and_record",
        "unsupported_references": "exclude_and_record",
        "reference_normalization": "outer_math_delimiters_and_contextual_clock_times",
        "reference_validation": "finite_parse_and_self_verification",
    }
    if source.sha256 != DATASET_SHA256:
        contract["source"] = source.identity()
    return contract


async def prepare_data(study):
    validate_grading_format(study)
    if study.data_manifest.exists():
        raise FileExistsError("Data manifest already exists; choose a new path to preserve its identity")
    source = dataset_source(study.dataset_path)
    contract = data_contract(
        study.prompt_instruction,
        study.prompt_max_tokens,
        study.reward_timeout_seconds,
        study.reasoning_required,
        source.sha256,
    )
    rows = read_rows(study.dataset_path)
    tok = tokenizer(str(study.prepared_model_path))
    records = [None] * len(rows)
    pending = iter(enumerate(rows))
    async with GraderPool(study.reward_workers, study.reward_outer_timeout_seconds, study.reward_retries) as pool:

        async def consume():
            for index, row in pending:
                prompt = row["prompt"] + ("\n\n" + study.prompt_instruction if study.prompt_instruction else "")
                token_ids = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}], add_generation_prompt=True, return_dict=False
                )
                record = {"id": row["id"], "prompt_tokens": len(token_ids)}
                if len(token_ids) > study.prompt_max_tokens:
                    record.update(included=False, reason="prompt_over_limit")
                else:
                    response = await pool.call(
                        {
                            "operation": "reference",
                            "answer": row["answers"][0],
                            "question": row["prompt"],
                            "timeout": study.reward_timeout_seconds,
                        }
                    )
                    included = response["status"] == "ok"
                    record.update(included=included, reason="accepted" if included else response["reason"])
                    if not included:
                        record["original_reference"] = row["answers"][0]
                        record["reference_failure"] = response.get("message", response["reason"])
                    else:
                        record["normalized_reference_sha256"] = digest(response["result"]["normalized_reference"])
                records[index] = record

        async with asyncio.TaskGroup() as tasks:
            for _ in range(study.reward_workers):
                tasks.create_task(consume())
    included = sum(record["included"] for record in records)
    if not included:
        raise ValueError("No trainable questions remain after deterministic preparation")
    if (
        data_contract(
            study.prompt_instruction,
            study.prompt_max_tokens,
            study.reward_timeout_seconds,
            study.reasoning_required,
            dataset_source(study.dataset_path).sha256,
        )
        != contract
    ):
        raise ValueError(
            "Data preparation contract changed while grading references; prepare again from frozen source"
        )
    body = {
        "format": 1,
        "contract": contract,
        "source_rows": len(rows),
        "included_rows": included,
        "excluded_rows": len(rows) - included,
        "records": records,
    }
    manifest = {**body, "sha256": digest(body)}
    atomic_write(study.data_manifest, (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode())
    return {key: manifest[key] for key in ("sha256", "source_rows", "included_rows", "excluded_rows")}


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    if manifest.get("format") != 1 or manifest.get("sha256") != digest(body):
        raise ValueError("Prepared data manifest failed integrity validation")
    contract = manifest["contract"]
    source = source_for_sha256(contract["source_sha256"])
    if contract["reward"] != reward_identity():
        raise ValueError("Prepared data belongs to a different source or grader; prepare it again")
    if source.sha256 != DATASET_SHA256 and contract.get("source") != source.identity():
        raise ValueError("Prepared data source metadata differs from its locked training release")
    return manifest


def prepared_rows(
    dataset_path,
    manifest_path,
    prompt_instruction,
    reward_timeout_seconds,
    prompt_max_tokens=None,
    reasoning_required=True,
):
    manifest = load_manifest(manifest_path)
    contract = manifest["contract"]
    expected = data_contract(
        prompt_instruction,
        contract["prompt_max_tokens"] if prompt_max_tokens is None else prompt_max_tokens,
        reward_timeout_seconds,
        reasoning_required,
        contract["source_sha256"],
    )
    if contract != expected:
        raise ValueError("Data preparation and run protocol disagree")
    rows = read_rows(dataset_path, expected_sha256=contract["source_sha256"])
    records = manifest["records"]
    if len(records) != len(rows) or any(row["id"] != record["id"] for row, record in zip(rows, records, strict=True)):
        raise ValueError("Data preparation changed the source question identity/order")
    included = [row for row, record in zip(rows, records, strict=True) if record["included"]]
    if (
        not included
        or len(included) != manifest["included_rows"]
        or len(rows) - len(included) != manifest["excluded_rows"]
    ):
        raise ValueError("Prepared dataset accounting is inconsistent")
    return included
