import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from exact_math import (
    INSTRUCTION,
    ExactMathConfig,
    ExactMathTaskset,
    verify_terminal_answer,
)
from ppo_loss import ppo_clip_loss
from prime_rl.trainer.rl.loss import LossInputs
from prepare_data import verify_reference_data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_hashes(path: Path) -> dict[str, str]:
    index_path = path / "model.safetensors.index.json"
    weights = (
        {"model.safetensors.index.json", *json.loads(index_path.read_text())["weight_map"].values()}
        if index_path.is_file()
        else {"model.safetensors"}
    )
    names = {
        "MODEL_INFO.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        *weights,
    }
    missing = sorted(name for name in names if not (path / name).is_file())
    if missing:
        raise ValueError(f"Missing model artifacts: {missing}")
    return {name: sha256(path / name) for name in sorted(names)}


def directory_hashes(path: Path) -> dict[str, str]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    return {str(item.relative_to(path)): sha256(item) for item in files}


def rows(path: Path):
    with path.open() as source:
        for line in source:
            yield json.loads(line)


def prompt_tokens(tokenizer, prompt: str) -> int:
    return len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": INSTRUCTION + prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.experiment_root / "data/manifest.json").read_text())
    for name, expected in manifest["hashes"].items():
        if sha256(args.experiment_root / "data" / name) != expected:
            raise ValueError(f"Hash mismatch for {name}")
    for name, expected in manifest["input_hashes"].items():
        if sha256(args.experiment_root / name) != expected:
            raise ValueError(f"Experiment input hash mismatch for {name}")
    spec = json.loads((args.experiment_root / "experiment.json").read_text())
    reference = verify_reference_data(
        Path(spec["reference_experiment"]) / "data",
        {name: list(rows(args.experiment_root / "data" / name)) for name in manifest["hashes"]},
    )
    if reference != manifest["reference_data"]:
        raise ValueError("The recorded comparison data provenance has changed")
    model_path = Path(manifest["model"])
    model_info = json.loads((model_path / "MODEL_INFO.json").read_text())
    if model_info["repository"] != spec["model"] or model_info["revision"] != spec["model_revision"]:
        raise ValueError("The model identity does not match the pinned 14B experiment")
    if model_hashes(model_path) != manifest["model_hashes"]:
        raise ValueError("Model snapshot hash mismatch")
    if sha256(Path(manifest["observer_archive"])) != manifest[
        "observer_archive_sha256"
    ]:
        raise ValueError("Opik observer archive hash mismatch")
    tokenizer_path = Path(manifest["tokenizer"])
    if directory_hashes(tokenizer_path) != manifest["tokenizer_hashes"]:
        raise ValueError("Tokenizer snapshot hash mismatch")
    expected_revisions = {
        "dapo": "31dd309567e3da778038cc87d868b6097a3ccf68",
        "math500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "aime24": "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
        "aime25": "a6ad95f611d72cf628a80b58bd0432ef6638f958",
        "aime26": "10b4e45b7a503075d4da8a0d57916a4f06ce6bd2",
        "intellect_math": "0c6d5c96f64bb8981220d8a280a8f998a8b1642f",
    }
    if manifest["dataset_revisions"] != expected_revisions:
        raise ValueError("Dataset revisions do not match the pinned experiment")
    expected_dapo = {
        "dapo_loaded": 17398,
        "dapo_kept": 17005,
        "dapo_duplicate_drops": 213,
        "dapo_conflicting_duplicate_groups": 5,
        "dapo_conflicting_duplicate_rows": 10,
        "dapo_overlap_drops": 167,
        "dapo_automatic_overlap_drops": 112,
        "dapo_audited_overlap_drops": 55,
        "dapo_length_drops": 8,
    }
    if any(manifest.get(key) != value for key, value in expected_dapo.items()):
        raise ValueError("Unexpected DAPO row counts")
    if manifest["dapo_overlap_drops"] != (
        manifest["dapo_automatic_overlap_drops"]
        + manifest["dapo_audited_overlap_drops"]
    ):
        raise ValueError("Overlap drop accounting is inconsistent")
    accounted = (
        manifest["dapo_kept"]
        + manifest["dapo_duplicate_drops"]
        + manifest["dapo_conflicting_duplicate_groups"]
        + manifest["dapo_overlap_drops"]
        + manifest["dapo_length_drops"]
    )
    if accounted != manifest["dapo_loaded"]:
        raise ValueError("DAPO row accounting is inconsistent")
    expected_eval = {
        "aime24": 30,
        "aime25": 30,
        "aime26": 30,
        "amc23": 40,
        "math500": 500,
        "minerva_math": 272,
        "olympiadbench": 675,
    }
    if manifest["eval_counts"] != expected_eval:
        raise ValueError("Unexpected evaluation row counts")
    tokenizer = AutoTokenizer.from_pretrained(
        args.experiment_root / "tokenizer", local_files_only=True
    )
    if tokenizer.eos_token != "<|im_end|>" or tokenizer.eos_token_id != 151645:
        raise ValueError("Tokenizer stop token is not <|im_end|>")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION + "What is 1+1?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not rendered.endswith("<think>\n\n</think>\n\n"):
        raise ValueError("Qwen3 non-thinking prefix is missing")
    train_count = 0
    train_max = 0
    train_prompts = set()
    train_source_ids = set()
    for row in rows(args.experiment_root / "data/train.jsonl"):
        actual = prompt_tokens(tokenizer, row["prompt"])
        if row["prompt_tokens"] != actual or actual > 1024:
            raise ValueError("Invalid training prompt token count")
        if not row["answers"] or row["prompt"] in train_prompts:
            raise ValueError("Invalid or duplicate training row")
        train_prompts.add(row["prompt"])
        train_source_ids.add(str(row["source_id"]))
        train_count += 1
        train_max = max(train_max, actual)
    if train_count != manifest["dapo_kept"]:
        raise ValueError("Training row count does not match the manifest")
    exclusions = json.loads(
        (args.experiment_root / "config/data_exclusions.json").read_text()
    )
    audited_ids = set(exclusions["audited_overlap_source_ids"])
    if train_source_ids & audited_ids:
        raise ValueError("Audited held-out variants remain in training data")
    eval_source_pairs = {
        (str(row["benchmark"]), str(row["source_id"]))
        for row in rows(args.experiment_root / "data/eval.jsonl")
    }
    invalid_exclusion_metadata = sorted(
        source_id
        for source_id, heldout in exclusions["audited_overlap_source_ids"].items()
        if (str(heldout["benchmark"]), str(heldout["eval_source_id"]))
        not in eval_source_pairs
    )
    if invalid_exclusion_metadata:
        raise ValueError("Audited overlap metadata does not resolve to evaluation rows")
    removed_overlaps = list(
        rows(args.experiment_root / "data/removed_overlaps.jsonl")
    )
    removed_audited_ids = {
        str(row["train_source_id"])
        for row in removed_overlaps
        if row.get("reason") == "audited_variant"
    }
    if (
        len(removed_overlaps) != manifest["dapo_overlap_drops"]
        or removed_audited_ids != audited_ids
    ):
        raise ValueError("Audited overlap removal record is incomplete")
    conflicts = list(
        rows(args.experiment_root / "data/removed_conflicting_duplicates.jsonl")
    )
    if (
        len(conflicts) != manifest["dapo_conflicting_duplicate_groups"]
        or any(len(record["rows"]) != 2 for record in conflicts)
    ):
        raise ValueError("Conflicting duplicate record is incomplete")
    eval_counts = {}
    eval_max = {}
    for row in rows(args.experiment_root / "data/eval.jsonl"):
        benchmark = row["benchmark"]
        limit = 2048 if benchmark in {"minerva_math", "olympiadbench"} else 1024
        actual = prompt_tokens(tokenizer, row["prompt"])
        if row["prompt_tokens"] != actual or actual > limit:
            raise ValueError("Invalid evaluation prompt token count")
        if not row["answers"]:
            raise ValueError("Evaluation row has no answer")
        eval_counts[benchmark] = eval_counts.get(benchmark, 0) + 1
        eval_max[benchmark] = max(eval_max.get(benchmark, 0), actual)
    if eval_counts != expected_eval or manifest["eval_length_drops"]:
        raise ValueError("Evaluation token profile does not match the experiment")
    taskset = ExactMathTaskset(
        ExactMathConfig(
            id="exact-math",
            dataset_path=str(args.experiment_root / "data/train.jsonl"),
            benchmark="dapo_train",
        )
    )
    tasks = taskset.load()
    for _ in range(3):
        next(tasks)
    trainer_logprobs = torch.tensor([-1.0, -0.8, -1.2], requires_grad=True)
    inputs = LossInputs(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=torch.tensor([-1.0, -1.0, -1.0]),
        ref_logprobs=None,
        advantages=torch.tensor([0.5, -0.5, 0.25]),
        loss_mask=torch.tensor([True, True, True]),
    )
    result = ppo_clip_loss(inputs, clip_eps=0.2)
    result.loss.backward()
    if (
        not torch.isfinite(result.loss)
        or not torch.isfinite(trainer_logprobs.grad).all()
    ):
        raise ValueError("Custom loss produced non-finite values")
    dummy_logprobs = torch.tensor([0.0], requires_grad=True)
    dummy = ppo_clip_loss(
        LossInputs(
            trainer_logprobs=dummy_logprobs,
            inference_logprobs=torch.tensor([0.0]),
            ref_logprobs=None,
            advantages=torch.tensor([0.0]),
            loss_mask=torch.tensor([False]),
        ),
        clip_eps=0.2,
    )
    dummy.loss.backward()
    if dummy.metrics or dummy.loss.item() != 0.0:
        raise ValueError("Custom loss dummy batch handling failed")
    reward_cases = [
        (r"Reasoning. \boxed{42}", "42", 1.0),
        ("Reasoning.\nFinal answer: 42", "42", 1.0),
        (r"The final answer is $\frac{1}{2}$", r"\frac{1}{2}", 1.0),
        ("Answer = x=2", "x=2", 1.0),
        ("<think>unfinished\nFinal answer: 42", "42", 0.0),
        ("We try 42, then conclude 41.", "42", 0.0),
        ("candidate 42\nFinal answer: 41", "42", 0.0),
        ("Final answer: 42\nCorrection: 41", "42", 0.0),
        (r"\boxed{41}" + "\nFinal answer: 42", "42", 1.0),
        (r"Final answer: $\frac{1}{1+\frac{1}{2}}$", r"\frac{2}{3}", 1.0),
        ("Final answer: 42.", "42", 1.0),
        ("Final answer: `42`", "42", 1.0),
        ("Final answer:", "42", 0.0),
        (r"Final answer: x + \boxed{42}", "42", 0.0),
        (r"Final answer: we tried \boxed{42} but it is 41", "42", 0.0),
        (r"We first obtain \boxed{42}, but the final result is 41.", "42", 0.0),
        (r"candidate \boxed{42}" + "\nCorrection: 41", "42", 0.0),
        (r"Therefore, \boxed{42}.", "42", 1.0),
        (r"The number of divisors is \(\boxed{42}\).", "42", 1.0),
        (r"The result is $\boxed{42}$.", "42", 1.0),
        ("Thus, the answer is:\n\\[\n\\boxed{42}\n\\]", "42", 1.0),
        ("Thus, the answer is:\n$$\n\\boxed{42}\n$$", "42", 1.0),
        ("\\[\n\\boxed{42}\n\\]\nCorrection: 41", "42", 0.0),
        ("\\[\n\\boxed{42}\n\\]\n```python\nprint(42)\n```", "42", 0.0),
    ]
    for reply, answer, expected_reward in reward_cases:
        if verify_terminal_answer(reply, answer, 5) != expected_reward:
            raise ValueError(f"Reward policy failed for {reply!r}")
    if max(verify_terminal_answer("Final answer: 42", answer, 5) for answer in ["41", "42"]) != 1.0:
        raise ValueError("Reward policy failed multiple-answer verification")
    print(
        json.dumps(
            {
                "data_hashes": "verified",
                "taskset": "verified",
                "tokenizer_eos_token_id": tokenizer.eos_token_id,
                "custom_loss": "verified",
                "train_prompt_tokens_max": train_max,
                "eval_prompt_tokens_max": eval_max,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
