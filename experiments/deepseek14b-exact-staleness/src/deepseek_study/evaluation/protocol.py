from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import digest, file_hash, read, tokenizer_identity, write


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Question(StrictModel):
    id: str = Field(min_length=1)
    benchmark: str = Field(min_length=1)
    question: str = Field(min_length=1)
    gold: str | list[str]
    repeats: int = Field(ge=1)
    meta: dict = Field(default_factory=dict)
    source_revision: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    row_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


def implementation_identity():
    return {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def upstream_identity():
    server = Path(find_spec("prime_rl.inference.server").origin)
    config = Path(find_spec("prime_rl.configs.inference").origin)
    return {"inference_config": file_hash(config), "inference_env": file_hash(server),
            "inference_patches": file_hash(server.parent / "patches.py"),
            "worker_init": file_hash(server.parent / "vllm/worker/__init__.py"),
            "worker": file_hash(server.parent / "vllm/worker/filesystem.py")}


class Protocol(StrictModel):
    format: Literal[1] = 1
    name: str
    questions: list[Question] = Field(min_length=1)
    response_tokens: int = Field(default=8192, ge=1)
    prompt_tokens: int = Field(default=2048, ge=1)
    instruction: str = r"Please reason step by step, and put your final answer within \boxed{}."
    temperature: float = Field(default=1.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    seed: int = Field(default=42, ge=0)
    questions_per_task: int = Field(default=16, ge=1, le=64)
    tokenizer: dict[str, str]
    grader: dict
    source: dict
    implementation: dict[str, str] = Field(default_factory=implementation_identity)
    upstream: dict[str, str] = Field(default_factory=upstream_identity)
    backend_versions: dict[str, str] = Field(default_factory=lambda: {
        "torch": "2.11.0+cu128", "transformers": "5.6.2", "vllm": "0.26.0+cu129",
    })

    @model_validator(mode="after")
    def validate_ids(self):
        if len({q.id for q in self.questions}) != len(self.questions):
            raise ValueError("Duplicate question IDs")
        if self.temperature == 0 and any(q.repeats != 1 for q in self.questions):
            raise ValueError("Greedy evaluation must have one repetition per question")
        return self

    @property
    def identity(self):
        return digest(self.model_dump())

    @classmethod
    def load(cls, path):
        return cls.model_validate(read(path))

    def samples(self, question_ids=None):
        selected = set(question_ids) if question_ids is not None else {q.id for q in self.questions}
        if not selected or not selected <= {q.id for q in self.questions}:
            raise ValueError("Invalid evaluation question subset")
        for question in self.questions:
            if question.id not in selected:
                continue
            for repeat in range(question.repeats):
                key = digest([question.id, repeat])
                # Independent of run, lag, checkpoint, task placement and retry order.
                seed = int(digest([self.seed, question.id, repeat])[:8], 16) % (2**31)
                yield key, question, repeat, seed

    def shards(self):
        for benchmark in sorted({q.benchmark for q in self.questions}):
            questions = [q.id for q in self.questions if q.benchmark == benchmark]
            for start in range(0, len(questions), self.questions_per_task):
                yield questions[start:start + self.questions_per_task]

    def validate_runtime(self, model, backend=True):
        if self.implementation != implementation_identity():
            raise ValueError("Evaluator source changed; use the frozen release or a new protocol")
        if self.upstream != upstream_identity():
            raise ValueError("Pinned PrimeRL inference implementation changed")
        if tokenizer_identity(model) != self.tokenizer:
            raise ValueError("Model tokenizer/config differs from the frozen evaluation protocol")
        if grader_identity() != self.grader:
            raise ValueError("Evaluation grader implementation or dependency versions changed")
        if backend:
            for package, expected in self.backend_versions.items():
                if version(package) != expected:
                    raise ValueError(f"Evaluation backend version differs: {package}")


def grader_identity():
    root = Path(__file__).parent
    return {
        "name": "math-sweep-final-v3.2",
        "files": {name: file_hash(root / name) for name in ("grader_v3.py", "extraction.py")},
        "dependencies": {name: version(name) for name in (
            "math-verify", "latex2sympy2-extended", "sympy", "antlr4-python3-runtime", "mpmath",
        )},
    }


def from_sweep(manifest_path, policy_path, model, destination, benchmarks=("math500", "aime24", "aime25")):
    manifest, policy = read(manifest_path), read(policy_path)
    if digest({k: v for k, v in manifest.items() if k != "experiment_hash"}) != manifest["experiment_hash"]:
        raise ValueError("Sweep manifest integrity failure")
    if policy["experiment_hash"] != manifest["experiment_hash"]:
        raise ValueError("Question policy belongs to another sweep")
    original = {p["id"]: p for p in manifest["problems"]}
    for section in ("corrections", "exclusions", "format_hints"):
        for key, item in policy[section].items():
            if key not in original or item["row_hash"] != original[key]["row_hash"]:
                raise ValueError(f"Question policy source mismatch: {key}")
            if "original_gold" in item and item["original_gold"] != original[key]["gold"]:
                raise ValueError(f"Question policy reference mismatch: {key}")
    questions = []
    allowed_hints = {"answer_labels", "unordered_sequence", "optional_units", "constant_case",
                     "component_labels", "symbol_aliases", "label_aliases", "component_units"}
    for q in manifest["problems"]:
        if q["benchmark"] not in benchmarks or q["id"] in policy["exclusions"]:
            continue
        fix = policy["corrections"].get(q["id"], {})
        hints = policy["format_hints"].get(q["id"], {})
        questions.append(Question(
            id=q["id"], benchmark=q["benchmark"], question=q["question"], gold=fix.get("gold", q["gold"]),
            repeats=q["attempts"], meta={**q["meta"], **fix.get("meta", {}),
                                       **{k: v for k, v in hints.items() if k in allowed_hints}},
            source_revision=q["source_revision"], source_sha256=q["source_sha256"], row_hash=q["row_hash"],
        ))
    if {q.benchmark for q in questions} != set(benchmarks):
        raise ValueError("Requested benchmark absent from manifest")
    protocol = Protocol(
        name="core-8k-v1", questions=questions, tokenizer=tokenizer_identity(model), grader=grader_identity(),
        source={"manifest_sha256": file_hash(manifest_path), "policy_sha256": file_hash(policy_path),
                "exclusions": policy["exclusions"], "corrections": policy["corrections"]},
    )
    if Path(destination).exists():
        raise FileExistsError(destination)
    write(destination, protocol.model_dump())
    return protocol
