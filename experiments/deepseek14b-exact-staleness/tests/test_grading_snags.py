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


@pytest.mark.parametrize(
    "answer,prediction,expected",
    [
        ("6:00", "6:00", 1),
        ("6:00", "06:00", 1),
        ("6:00", "6:01", 0),
        ("6:00", "6", 0),
        ("6:00", "6:0", 0),
        ("6:00", "6:60", 0),
        ("6:00", "24:00", 0),
        ("6:00", "6:00 or 7:00", 0),
        ("6:00", r"\text{6:00}", 1),
        ("6:00", "6:00 PM", 1),
        ("6:00", "6:00 AM", 1),
        ("6:00", "18:00", 1),
        ("6:00 PM", "6:00", 0),
        ("18:00", "6:00 AM", 0),
        ("18:00", "6:00", 0),
        ("6:00", r"6:00\ \text{pm}", 1),
        (r"8\!:\!34 \text{p.m.}", "20:34", 1),
        (r"05\!:\!00", "5:00", 1),
        ("6:00", r"6\quad:\,00\;\text{p.m.}", 1),
        ("6:00 PM", "18:00", 1),
        ("6:00 PM", "6:00 AM", 0),
        ("12:00 AM", "00:00", 1),
        ("12:00 PM", "12:00 PM", 1),
    ],
)
def test_clock_question_uses_time_semantics_instead_of_division(answer, prediction, expected):
    question = "A clock is running slowly. What is the actual time?"
    assert grade("</think>\\boxed{" + prediction + "}", answer, False, "zero", 5, question=question) == expected


@pytest.mark.parametrize("answer", ["6:0", "6:60", "24:00", "0:00 AM", "13:00 PM", "6:00 or 7:00"])
def test_malformed_clock_references_are_excluded_explicitly(answer):
    with pytest.raises(ReferenceRejected, match="Clock reference"):
        parse_gold(answer, 5, question="What is the actual time on the clock?")


@pytest.mark.parametrize(
    "answer,prediction,question",
    [
        ("3:4", r"\frac{3}{4}", "What is the ratio of boys to girls?"),
        ("6:30", r"\frac{1}{5}", "What is the ratio of the two clock hand speeds?"),
        ("6:30", r"\frac{1}{5}", ""),
    ],
)
def test_ratio_answers_keep_original_mathematical_semantics(answer, prediction, question):
    assert grade("</think>\\boxed{" + prediction + "}", answer, False, "zero", 5, question=question) == 1


@pytest.mark.parametrize(
    "answer,prediction,question",
    [
        ("6:00", "06:00", "The clock in Sri's car gains time. What is the actual time?"),
        ("10:15", "10:15", "The watch reads 10:00. What is the exact time now?"),
        ("11:00", "11:00", "They leave at 8:30 AM and 9:00 AM. At what time in the morning do they meet?"),
        ("1:36", "1:36", "At what time P.M. should the candles be lighted?"),
        ("4:50", "4:50", "At 7:10 in the morning, he mistakenly believes the time is ___ hours ___ minutes."),
        ("12:30", "12:30", "Theo's watch reads 12:00. What time does Leo think it is?"),
        ("9:24", "9:24", "The runners start at 8:00. What is the earliest time they meet again?"),
        (r"\text{4:10 P.M.}", "16:10", "She begins at 7:25 A.M. When will her work end?"),
        ("5:05(PM)", "17:05", "Candice starts driving at 5:00 PM. What time is it when she gets home?"),
        (r"7:55 \text{ p.m.}", "19:55", "Starting at 1:00 p.m., Jorge watches movies. When will he finish?"),
    ],
)
def test_clock_formats_and_context_from_locked_dataset(answer, prediction, question):
    assert grade("</think>\\boxed{" + prediction + "}", answer, False, "zero", 5, question=question) == 1


@pytest.mark.parametrize(
    "answer,question",
    [
        ("7:23 and 7:53", "Find the times when the clock hands are separated by 84 degrees."),
        ("12:26:40", "Cyclists start at noon. When will they next meet?"),
        ("4:26.8", "The watch hands exchanged places. When did he leave?"),
        (r"4:45\frac{3}{11}", "The hands of a clock are opposite. Determine the exact time now."),
        ("14:24", "How long does it take to row? Express your answer in the format mm:ss."),
        ("1:2", "In the soccer match between North and South, what was the final score?"),
    ],
)
def test_unsupported_structured_colon_references_are_not_treated_as_ratios(answer, question):
    with pytest.raises(ReferenceRejected):
        parse_gold(answer, 5, question=question)
