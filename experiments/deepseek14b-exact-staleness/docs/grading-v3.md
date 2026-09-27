# Grading policy v3

The new policy fixes two observed failures: the mathematical parser treated the clock reference `6:00` as division by zero, and the grader required a reasoning closing tag from the base Qwen2.5 model even though its native prompt does not open a reasoning section. The reward remains binary correctness of the final complete boxed answer.

## Final-answer boundaries

`reasoning_required` defaults to `true`. Native DeepSeek-R1-Distill and Qwen3 thinking prompts retain that setting because their opening `<think>` can be supplied by the prompt rather than generated tokens. Base Qwen2.5 uses `false`, allowing a plain boxed answer.

Regardless of the setting, generated `<think>` sections must close. When a closing tag is present, only the text after the final `</think>` is graded. Unfinished or reopened reasoning cannot earn a reward from a box inside it. The existing truncation policy still applies.

## Clock answers and ratios

Clock handling uses the original question as context. Explicit clock, watch, time-of-day, or scheduling context selects strict clock parsing; ratio or proportion questions keep mathematical ratio semantics. A colon alone does not establish a clock answer. For example, `6:30` can represent a mathematical ratio without clock context, while the failed car-clock question is interpreted as a time.

Supported clock answers contain an hour and exactly two minute digits, with an optional AM/PM qualifier. Hours and minutes must be valid. Safe LaTeX text wrappers and spacing commands are accepted, including `\!`, `\,`, `\;`, `\:`, escaped spaces, and named spacing such as `\quad`. These changes accept the corpus references `8\!:\!34 \text{p.m.}` and `05\!:\!00`.

The actual failed 14B response boxed `6:00\ \text{pm}`; the failed Qwen3 response boxed `6:00`. Both are accepted against the clock reference `6:00` under the new policy.

An unqualified 12-hour reference specifies only a clock-face hour and minute. It therefore accepts a matching AM/PM-qualified or 24-hour prediction without claiming that the period was verified. A reference with an explicit period or an unambiguous 24-hour value requires the corresponding period. For example, `6:00 PM` matches `18:00` and rejects `6:00 AM` or an unqualified `6:00`. This is a documented limitation of underspecified references, rather than an inference about the story's intended period.

Multiple clock times, seconds, fractional minutes, explicit `mm:ss` durations, and match-score answers do not have implemented structured comparators. Such references are excluded and recorded rather than passed to mathematical division. This does not broaden the parser to every colon-separated format.

## Invalid mathematics and worker failures

Reference preparation now verifies that parsed values can be compared with themselves. Undefined `NaN` and complex infinity values are rejected, including when nested in a structured answer. Supported mathematical infinity notation remains valid where the verifier supports it. Existing tuple/set, interval, symbol-case, and large-integer safeguards remain in force.

Recognized mathematical content failures in a prediction yield reward zero with an explicit diagnostic reason. Invalid references are rejected during preparation. Infrastructure failures, unexpected errors, and parser/comparison timeouts remain worker failures subject to bounded retries; persistent failures stop the run. They are not silently converted into incorrect answers.

## Validation and migration

Preparing all 37,713 locked source questions with this policy includes 37,696 and records 17 exclusions. Seven exclusions are new relative to v2. The complete preparation manifest records question identities and exclusion reasons.

Regression tests cover the observed failures and boundary cases. Successful reference preparation and replay of saved responses do not establish the grader's overall accuracy or guarantee that every future prediction is supported.

The policy changes the reward identity and prepared-data contract. Future runs require a fresh manifest and frozen release. They must not reuse v2 manifests or resume old checkpoints as though the reward protocol were unchanged. Existing reports and stopped-run outputs remain historical evidence.
