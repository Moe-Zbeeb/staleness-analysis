import pytest

from deepseek_study.rewards import ReferenceRejected, grade, last_complete_box, normalize_reference, parse_gold


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
    import deepseek_study.rewards as rewards

    def fail(*args, **kwargs):
        raise RuntimeError("worker failure")

    monkeypatch.setattr(rewards, "verify", fail)
    with pytest.raises(RuntimeError, match="worker failure"):
        grade("</think>\\boxed{4}", "4", False, "zero", 5)
