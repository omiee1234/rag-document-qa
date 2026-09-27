from ragqa.evaluate import (
    faithfulness_overlap,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_all_found():
    assert recall_at_k(["a", "b", "c"], ["a"]) == 1.0


def test_recall_at_k_none_found():
    assert recall_at_k(["x", "y"], ["a"]) == 0.0


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], ["a"]) == 1 / 3


def test_reciprocal_rank_first_hit():
    assert reciprocal_rank(["a", "b"], ["a"]) == 1.0


def test_reciprocal_rank_second_hit():
    assert reciprocal_rank(["b", "a"], ["a"]) == 0.5


def test_reciprocal_rank_no_hit():
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_faithfulness_overlap_true_when_answer_grounded():
    answer = "Employees accrue 18 days of PTO per year."
    context = ["Full-time employees accrue 18 days of paid time off per calendar year."]
    assert faithfulness_overlap(answer, context) is True


def test_faithfulness_overlap_false_when_answer_unrelated():
    answer = "The quarterly revenue grew by forty percent this year."
    context = ["Full-time employees accrue 18 days of paid time off per calendar year."]
    assert faithfulness_overlap(answer, context) is False
