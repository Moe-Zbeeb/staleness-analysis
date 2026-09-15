import json
import re
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf

INSTRUCTION = "Solve the following math problem. Explain your reasoning. End with either \\boxed{...} or a final line `Final answer: ...`.\n\n"
FINAL_ANSWER = re.compile(
    r"(?i)^(?:(?:thus|therefore|hence)[,:]?\s+)?(?:the\s+)?"
    r"(?:final\s+)?answer\s*(?:is\s*[:=]?|[:=])\s*"
    r"(?P<answer>.+?)\s*(?:[.!])?\s*$"
)
CUE = re.compile(r"(?i)^(?:thus|therefore|hence)[,:]?\s+(.+?)\s*$")
MATH_CLOSERS = ("$$", "$", r"\)", r"\]", "`")
TERMINAL_WRAPPER = re.compile(r"(?:\$\$|\$|\\\)|\\\]|`)?[.!]?\s*$")


def visible_completion(reply: str) -> str | None:
    lowered = reply.lower()
    opening = lowered.rfind("<think>")
    closing = lowered.rfind("</think>")
    if opening > closing:
        return None
    if closing >= 0:
        return reply[closing + len("</think>") :].strip()
    return reply.strip()


def strip_math_delimiters(value: str) -> str:
    pairs = (
        ("$$", "$$"),
        ("$", "$"),
        (r"\(", r"\)"),
        (r"\[", r"\]"),
        ("`", "`"),
    )
    for opening, closing in pairs:
        if value.startswith(opening) and value.endswith(closing):
            return value[len(opening) : -len(closing)].strip()
    return value


def exact_box(value: str) -> str | None:
    candidate = value.strip()
    if candidate.endswith((".", "!")):
        candidate = candidate[:-1].rstrip()
    cue = CUE.fullmatch(candidate)
    if cue is not None:
        candidate = cue.group(1).strip()
    candidate = strip_math_delimiters(candidate)
    extracted = vf.extract_boxed_answer(candidate, strict=True)
    if extracted is None or candidate != rf"\boxed{{{extracted}}}":
        return None
    return rf"\boxed{{{extracted}}}"


def terminal_box(value: str) -> str | None:
    candidate = TERMINAL_WRAPPER.sub("", value.strip()).rstrip()
    extracted = vf.extract_boxed_answer(candidate, strict=True)
    if extracted is None:
        return None
    boxed = rf"\boxed{{{extracted}}}"
    if not candidate.endswith(boxed):
        return None
    return boxed


def verify_terminal_answer(reply: str, answer: str, timeout_seconds: int) -> float:
    visible = visible_completion(reply)
    if not visible:
        return 0.0
    lines = [line.strip() for line in visible.splitlines() if line.strip()]
    if lines:
        closing = lines[-1]
        if closing.endswith((".", "!")):
            closing = closing[:-1].rstrip()
        if closing in MATH_CLOSERS:
            lines.pop()
    terminal = lines[-1] if lines else ""
    match = FINAL_ANSWER.fullmatch(terminal)
    if match is not None:
        candidate = match.group("answer").strip()
        if not candidate or len(candidate) > 512:
            return 0.0
        boxed = exact_box(candidate)
        if boxed is not None:
            candidate = boxed
        elif r"\boxed" in candidate:
            return 0.0
        else:
            candidate = rf"\boxed{{{strip_math_delimiters(candidate)}}}"
        return vf.verify_boxed_math_answer(
            candidate,
            answer,
            timeout_seconds=timeout_seconds,
        )
    candidate = terminal_box(terminal)
    if candidate is None:
        return 0.0
    return vf.verify_boxed_math_answer(
        candidate,
        answer,
        timeout_seconds=timeout_seconds,
    )


class ExactMathData(vf.TaskData):
    answers: list[str]


class ExactMathTaskConfig(vf.TaskConfig):
    math_verify_timeout: int = 5


class ExactMathTask(vf.Task[ExactMathData, vf.State, ExactMathTaskConfig]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return max(
            verify_terminal_answer(
                trace.last_reply,
                answer,
                self.config.math_verify_timeout,
            )
            for answer in self.data.answers
        )


class ExactMathConfig(vf.TasksetConfig):
    dataset_path: str
    benchmark: str
    task: ExactMathTaskConfig = ExactMathTaskConfig()


class ExactMathTaskset(vf.Taskset[ExactMathTask, ExactMathConfig]):
    def load(self) -> Iterator[ExactMathTask]:
        matched = 0
        with Path(self.config.dataset_path).open() as source:
            for row in source:
                record = json.loads(row)
                if record["benchmark"] != self.config.benchmark:
                    continue
                yield ExactMathTask(
                    ExactMathData(
                        idx=matched,
                        prompt=INSTRUCTION + record["prompt"],
                        answers=[str(answer) for answer in record["answers"]],
                    ),
                    self.config.task,
                )
                matched += 1
        if matched == 0:
            raise ValueError(f"No rows found for benchmark {self.config.benchmark!r}")


__all__ = ["ExactMathTaskset"]
