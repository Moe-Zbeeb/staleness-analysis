import pytest

from deepseek_study.dataset.rewards import ReferenceRejected, grade, last_complete_box, normalize_reference, parse_gold


@pytest.mark.parametrize(
    "response,answer,expected",
    [
        ("work \\boxed{4}", "4", 0),
        ("work \\boxed{4}</think>final \\boxed{5}", "4", 0),
        ("work</think>final \\boxed{\\frac{1}{2}}", "0.5", 1),
        ("work</think>4", "4", 0),
        ("work</think>\\boxed{4}<think>unfinished", "4", 0),
        ("work</think>\\boxed{4", "4", 0),
    ],
)
def test_reward_requires_closed_reasoning_and_correct_final_box(response, answer, expected):
    assert grade(response, answer, False, "zero", 5) == expected


def test_truncation_policy_is_explicit():
    response = "</think>\\boxed{4}"
    assert grade(response, "4", True, "zero", 5) == 0
    assert grade(response, "4", True, "grade_final", 5) == 1


def test_last_complete_box_does_not_use_reasoning_box_or_incomplete_suffix():
    assert grade("\\boxed{5}</think>\\boxed{4} later \\boxed{", "4", True, "grade_final", 5) == 1
    assert last_complete_box("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert last_complete_box("\\boxed{\\{1,2\\}}") == "\\{1,2\\}"


@pytest.mark.parametrize("text", ["$4$", "\\[4\\]", "\\(4\\)", "$$4$$"])
def test_reference_normalization_only_removes_outer_math_delimiters(text):
    assert normalize_reference(text) == "4"
    assert grade("</think>\\boxed{4}", text, False, "zero", 5) == 1


def test_partial_root_list_reference_is_rejected_instead_of_accepting_one_root():
    with pytest.raises(ReferenceRejected):
        parse_gold(r"$\frac{1}{2}, \frac{-1 \pm \sqrt{13}}{4}$", 5)


def test_comparison_failure_propagates_instead_of_becoming_a_wrong_answer(monkeypatch):
    import deepseek_study.dataset.rewards as rewards

    def fail(*args, **kwargs):
        raise RuntimeError("worker failure")

    monkeypatch.setattr(rewards, "verify", fail)
    with pytest.raises(RuntimeError, match="worker failure"):
        grade("</think>\\boxed{4}", "4", False, "zero", 5)


@pytest.mark.parametrize(
    "answer,prediction,expected",
    [
        ("(1997,0)", r"\{1997,0\}", 0),
        ("(1997,0)", "{1997,0}", 0),
        (r"\{1997,0\}", "(1997,0)", 0),
        ("(1997,0)", "(1997,0)", 1),
        ("(1997,0)", "(0,1997)", 0),
        (r"\{1,2\}", r"\{2,1\}", 1),
        (r"\{1,2\}", r"\{1\}", 0),
        ("(1,(2,3))", r"(1,\{2,3\})", 0),
        ("A", "a", 0),
        ("A+B", "a+b", 0),
        ("A+A", "2A", 1),
        ("A-a", "0", 0),
        ("0", "A-a", 0),
        ("A-A", "0", 1),
        (r"\text{ABC}", "ABC", 1),
        ("1000000000000000001", "1000000000000000000", 0),
        (r"\frac{1}{2}", "0.5", 1),
        ("(1,2)", r"\left(1,2\right)", 1),
    ],
)
def test_structure_and_case_safeguards(answer, prediction, expected):
    assert grade("</think>\\boxed{" + prediction + "}", answer, False, "grade_final", 8) == expected


@pytest.mark.parametrize(
    "response,reasoning_required,expected",
    [
        (r"Final answer: \boxed{4}", False, 1),
        (r"Final answer: \boxed{4}", True, 0),
        (r"<think>work \boxed{4}", False, 0),
        (r"<think><think>work</think>\boxed{4}", False, 0),
        (r"</think><think><think>work</think>\boxed{4}", False, 0),
        (r"<think>work \boxed{5}</think>\boxed{4}", False, 1),
        (r"<think>work \boxed{4}</think>\boxed{5}", False, 0),
        (r"\boxed{4}</think>no final answer", False, 0),
        (r"</think>\boxed{4}<think>more work", False, 0),
        (r"</think>\boxed{4}<think>more work</think>\boxed{5}", False, 0),
    ],
)
def test_native_reasoning_requirement_and_explicit_thinking_boundaries(response, reasoning_required, expected):
    assert grade(response, "4", False, "zero", 5, reasoning_required=reasoning_required) == expected


@pytest.mark.parametrize("answer", [r"\frac{1}{0}", r"\frac{0}{0}", r"(1,\frac{1}{0})", "6:00"])
def test_undefined_references_are_rejected_before_training(answer):
    with pytest.raises(ReferenceRejected, match="undefined"):
        parse_gold(answer, 5)


@pytest.mark.parametrize("prediction", [r"\frac{1}{0}", r"\frac{0}{0}", r"(1,\frac{1}{0})", "6:00"])
def test_undefined_prediction_is_logged_as_wrong_instead_of_crashing(prediction):
    from deepseek_study.dataset.rewards import grade_result

    result = grade_result("</think>\\boxed{" + prediction + "}", "4", False, "zero", 5)
    assert result["reward"] == 0
    assert result["reason"] in {"undefined_prediction", "unsupported_prediction"}


@pytest.mark.parametrize("answer", [r"\infty", r"(-\infty,1)"])
def test_supported_infinity_notation_remains_verifiable(answer):
    assert grade("</think>\\boxed{" + answer + "}", answer, False, "zero", 5) == 1


def test_reference_validation_checks_verifiability_not_only_parsing(monkeypatch):
    import deepseek_study.dataset.rewards as rewards

    rewards.parse_gold.cache_clear()
    monkeypatch.setattr(rewards, "verify", lambda *args, **kwargs: False)
    with pytest.raises(ReferenceRejected, match="against itself"):
        rewards.parse_gold("23", 5)
    rewards.parse_gold.cache_clear()


@pytest.mark.parametrize("stage", ["parse", "compare", "case"])
@pytest.mark.parametrize("failure", [RuntimeError("worker failure"), OSError("worker I/O failure"), MemoryError()])
def test_infrastructure_failures_never_become_incorrect_rewards(monkeypatch, stage, failure):
    import deepseek_study.dataset.rewards as rewards

    rewards.parse_gold("4", 5)

    def fail(*args, **kwargs):
        raise failure

    if stage == "parse":
        monkeypatch.setattr(rewards, "parse", fail)
    elif stage == "compare":
        monkeypatch.setattr(rewards, "verify", fail)
    else:
        monkeypatch.setattr(rewards, "parse", lambda *args, **kwargs: rewards.parse_gold("4", 5))
        monkeypatch.setattr(rewards, "verify", lambda *args, **kwargs: True)
        monkeypatch.setattr(rewards, "case_preserved", fail)
    with pytest.raises(type(failure)):
        rewards.grade("</think>\\boxed{A}", "4", False, "zero", 5)


def test_prediction_timeout_is_retried_and_explicitly_unverified(monkeypatch):
    import deepseek_study.dataset.rewards as rewards
    from math_verify.errors import TimeoutException

    rewards.parse_gold("4", 5)

    def fail(*args, **kwargs):
        raise TimeoutException("comparison timeout")

    monkeypatch.setattr(rewards, "verify", fail)
    result = rewards.grade_result("</think>\\boxed{4}", "4", False, "zero", 5)
    assert result["reward"] == 0
    assert result["reason"] == "prediction_verification_timeout"
    assert result["verification_status"] == "unverified"
    assert [item["timeout_seconds"] for item in result["verification_timeouts"]] == [5, 20]


def test_signal_thread_configuration_failure_is_not_misclassified(monkeypatch):
    import deepseek_study.dataset.rewards as rewards

    rewards.parse_gold("4", 5)

    def fail(*args, **kwargs):
        raise ValueError("signal only works in main thread of the main interpreter")

    monkeypatch.setattr(rewards, "verify", fail)
    with pytest.raises(ValueError, match="signal"):
        rewards.grade("</think>\\boxed{4}", "4", False, "zero", 5)


def test_content_comparison_failure_is_explicit_incorrect_prediction(monkeypatch):
    import deepseek_study.dataset.rewards as rewards

    rewards.parse_gold("4", 5)

    def fail(*args, **kwargs):
        raise ValueError("invalid mathematical operand")

    monkeypatch.setattr(rewards, "verify", fail)
    result = rewards.grade_result("</think>\\boxed{4}", "4", False, "zero", 5)
    assert result["reward"] == 0
    assert result["reason"] == "unsupported_prediction"


def test_reference_timeout_still_fails_closed(monkeypatch):
    import deepseek_study.dataset.rewards as rewards
    from math_verify.errors import TimeoutException

    def fail(*args, **kwargs):
        raise TimeoutException("reference timeout")

    monkeypatch.setattr(rewards, "parse_gold", fail)
    with pytest.raises(TimeoutException):
        rewards.grade_result("</think>\\boxed{4}", "4", False, "zero", 5)


def test_prediction_retry_can_recover_without_regeneration(monkeypatch):
    import deepseek_study.dataset.rewards as rewards
    from math_verify.errors import TimeoutException

    calls = []

    def verify(gold, boxed, answer, timeout):
        calls.append((boxed, answer, timeout))
        if len(calls) == 1:
            raise TimeoutException("comparison timeout")
        return {"reward": 1.0, "reason": "correct", "extracted": boxed, "policy": rewards.POLICY}

    monkeypatch.setattr(rewards, "verify_prediction", verify)
    result = rewards.grade_result("</think>\\boxed{4}", "4", False, "zero", 5)
    assert result["reward"] == 1
    assert calls == [("4", "4", 5), ("4", "4", 20)]
    assert len(result["verification_timeouts"]) == 1


def test_production_large_exponent_answer_is_bounded():
    from deepseek_study.dataset.rewards import grade_result

    result = grade_result(
        r"</think>\boxed{2^{2019} - \left(1 + e^{-\frac{1}{e}}\right)^{2019}}",
        "-1",
        False,
        "grade_final",
        1,
    )
    assert result["reward"] == 0
    assert result["reason"] in {"not_verified_correct", "prediction_verification_timeout"}
