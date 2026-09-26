import pytest

from deepseek_study.dataset.rewards import ReferenceRejected, grade, parse_gold


@pytest.mark.parametrize(
    "answer,prediction,expected",
    [
        ("2050312", "2050313", 0),
        ("(1,2)", "[1,2]", 0),
        ("[1,2)", "[1,2]", 0),
        ("x=1,y=2", "x=2,y=1", 0),
        ("2", r"2+\unsupported{z}", 0),
        ("2", r"2\quad\text{and }3", 0),
        ("2", "2+?", 0),
        (r"\begin{pmatrix}1&2\end{pmatrix}", r"\begin{pmatrix}1\\2\end{pmatrix}", 0),
        (r"\begin{pmatrix}1&0\\0&1\end{pmatrix}", r"\begin{vmatrix}1&0\\0&1\end{vmatrix}", 0),
        ("2050312", "2,050,312", 1),
        ("i", r"\sqrt{-1}", 0),
        ("120", "5!", 1),
        ("10", r"\binom{5}{2}", 1),
        ("x+y=2", "y=2-x", 1),
    ],
)
def test_reported_grading_categories_have_explicit_probes(answer, prediction, expected):
    assert grade("</think>\\boxed{" + prediction + "}", answer, False, "grade_final", 8) == expected


def test_unsupported_subscripted_energy_reference_is_explicitly_rejected():
    with pytest.raises(ReferenceRejected):
        parse_gold("E_k", 8)
