import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    model_path: Path
    dataset_path: Path
    data_manifest: Path
    prepared_model_path: Path
    output_dir: Path
    metrics_mirror_root: Path | None = None
    prompts_per_update: int = Field(ge=1)
    responses_per_prompt: int = Field(ge=2)
    prompt_max_tokens: int = Field(ge=1)
    response_max_tokens: int = Field(ge=1)
    max_steps: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    lr_warmup_steps: int = Field(ge=0)
    lr_decay_steps: int = Field(ge=0)
    min_learning_rate: float = Field(ge=0)
    weight_decay: float = Field(ge=0)
    adam_beta1: float = Field(ge=0, lt=1)
    adam_beta2: float = Field(ge=0, lt=1)
    adam_epsilon: Literal[1e-8]
    max_grad_norm: float = Field(gt=0)
    clip_epsilon: float = Field(gt=0, lt=1)
    advantage_normalization: Literal["centered", "population_std", "sample_std"]
    advantage_epsilon: float = Field(gt=0)
    temperature: Literal[1.0]
    loss_reduction: Literal["global_token_mean"]
    reference_kl_coefficient: Literal[0.0]
    updates_per_cohort: Literal[1]
    prompt_instruction: str
    truncated_reward: Literal["zero", "grade_final"]
    seed: int = Field(ge=0)
    checkpoint_interval: int = Field(ge=1)
    checkpoint_keep_last: int = Field(default=4, ge=1)
    checkpoint_keep_interval: int = Field(default=100, ge=1)
    trainer_gpus: int = Field(ge=1)
    inference_gpus: int = Field(ge=1)
    inference_tensor_parallel: Literal[1, 2, 4, 8]
    rollout_concurrency: int = Field(ge=1)
    trainer_micro_batch_size: Literal[1]
    trainer_attention: Literal["fa2", "fa3", "fa4"]
    trainer_compile: bool
    optimizer_cpu_offload: bool
    activation_checkpointing: bool
    activation_cpu_offload: bool = False
    lm_head_chunk_size: int = Field(default=8192, ge=1)
    inference_memory_utilization: float = Field(gt=0, lt=1)
    inference_max_sequences: int = Field(default=16, ge=1)
    lag: int = Field(ge=0, strict=True)
    timeout_seconds: int = Field(default=1800, ge=1)
    generation_timeout_seconds: int = Field(default=86400, ge=1)
    reward_timeout_seconds: int = Field(default=8, ge=1)
    reward_outer_timeout_seconds: int = Field(default=10, ge=1)
    reward_workers: int = Field(default=4, ge=1)
    reward_retries: int = Field(default=1, ge=0, le=3)
    inference_port: int = Field(default=8000, ge=1024, le=44000)

    @model_validator(mode="after")
    def validate_layout(self):
        if self.max_steps <= self.lag:
            raise ValueError(
                "The total update budget must include at least one exact-staleness update after bootstrap"
            )
        if self.reward_outer_timeout_seconds <= self.reward_timeout_seconds:
            raise ValueError("The outer grading deadline must exceed the internal timeout")
        if self.activation_cpu_offload and not self.activation_checkpointing:
            raise ValueError("Activation CPU offload requires activation checkpointing")
        if self.inference_gpus % self.inference_tensor_parallel:
            raise ValueError("Inference GPU count must be divisible by tensor parallelism")
        if self.rollout_concurrency < self.responses_per_prompt:
            raise ValueError(
                "PrimeRL request concurrency must allow one complete response group; use inference_max_sequences to bound active GPU sequences"
            )
        if self.lr_warmup_steps >= self.max_steps:
            raise ValueError("Learning-rate warmup must end before the training budget")
        if self.lr_warmup_steps + self.lr_decay_steps > self.max_steps:
            raise ValueError("Learning-rate warmup and decay exceed the training budget")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("Minimum learning rate exceeds the peak")
        if self.sequence_length > 131072:
            raise ValueError("Sequence length exceeds the pinned model context")
        inputs = (self.model_path, self.dataset_path, self.prepared_model_path, self.data_manifest)
        if self.metrics_mirror_root is not None:
            mirror = self.metrics_mirror_root.resolve() / self.output_dir.name
            if mirror.is_relative_to(self.output_dir.resolve()) or self.output_dir.resolve().is_relative_to(mirror):
                raise ValueError("Metric mirror and run directory must be separate")
            if any(mirror.is_relative_to(path.resolve()) or path.resolve().is_relative_to(mirror) for path in inputs):
                raise ValueError("Metric mirror must not overlap input assets")
        if any(
            self.output_dir.resolve().is_relative_to(path.resolve())
            for path in (self.model_path, self.prepared_model_path)
        ):
            raise ValueError("Outputs must not be written inside model assets")
        if any(
            self.output_dir.resolve() == path.resolve() or path.resolve().is_relative_to(self.output_dir.resolve())
            for path in inputs
        ):
            raise ValueError("The run directory must not overwrite an input asset")
        return self

    @property
    def response_batch_size(self):
        return self.prompts_per_update * self.responses_per_prompt

    @property
    def sequence_length(self):
        return self.prompt_max_tokens + self.response_max_tokens

    def fingerprint(self):
        content = self.model_dump(mode="json", exclude={"output_dir", "metrics_mirror_root"})
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    @classmethod
    def read(cls, path):
        return cls.model_validate_json(Path(path).read_text())
