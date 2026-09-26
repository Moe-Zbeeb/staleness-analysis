import random
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf

from deepseek_study.dataset.prepare import prepared_rows
from deepseek_study.dataset.grading import GraderFailure, get_pool
from deepseek_study.dataset.rewards import completion_tokens, tokenizer

__all__ = ["DeepScaleRTaskset"]


class DeepScaleRData(vf.TaskData):
    question_id: str
    answer: str
    tokenizer_path: str
    truncated_reward: Literal["zero", "grade_final"]
    reward_timeout_seconds: int
    reward_outer_timeout_seconds: int
    reward_workers: int
    reward_retries: int


class DeepScaleRTask(vf.Task[DeepScaleRData]):
    @property
    def key(self):
        return self.data.question_id

    @vf.reward(weight=1.0)
    async def correctness(self, trace: vf.Trace) -> float:
        raw = tokenizer(self.data.tokenizer_path).decode(completion_tokens(trace), skip_special_tokens=False)
        pool = get_pool(self.data.reward_workers, self.data.reward_outer_timeout_seconds, self.data.reward_retries)
        response = await pool.call(
            {
                "operation": "grade",
                "arguments": {
                    "raw_completion": raw,
                    "answer": self.data.answer,
                    "truncated": trace.is_truncated,
                    "truncated_reward": self.data.truncated_reward,
                    "timeout": self.data.reward_timeout_seconds,
                },
            }
        )
        trace.info["study_grading"] = response
        if response["status"] != "ok":
            raise GraderFailure("A prepared reference is no longer supported by the grader")
        return response["result"]["reward"]


class DeepScaleRConfig(vf.TasksetConfig):
    dataset_path: str
    data_manifest: str
    tokenizer_path: str
    prompt_instruction: str
    truncated_reward: Literal["zero", "grade_final"]
    reward_timeout_seconds: int
    reward_outer_timeout_seconds: int
    reward_workers: int
    reward_retries: int
    seed: int


class DeepScaleRTaskset(vf.Taskset[DeepScaleRTask, DeepScaleRConfig]):
    def load(self):
        rows = prepared_rows(
            Path(self.config.dataset_path),
            Path(self.config.data_manifest),
            self.config.prompt_instruction,
            self.config.reward_timeout_seconds,
        )
        random.Random(self.config.seed).shuffle(rows)
        return [
            DeepScaleRTask(
                DeepScaleRData(
                    idx=index,
                    question_id=row["id"],
                    prompt=row["prompt"]
                    + ("\n\n" + self.config.prompt_instruction if self.config.prompt_instruction else ""),
                    answer=row["answers"][0],
                    tokenizer_path=self.config.tokenizer_path,
                    truncated_reward=self.config.truncated_reward,
                    reward_timeout_seconds=self.config.reward_timeout_seconds,
                    reward_outer_timeout_seconds=self.config.reward_outer_timeout_seconds,
                    reward_workers=self.config.reward_workers,
                    reward_retries=self.config.reward_retries,
                ),
                self.config.task,
            )
            for index, row in enumerate(rows)
        ]
