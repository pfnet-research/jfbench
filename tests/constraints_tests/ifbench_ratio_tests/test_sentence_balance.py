from jfbench.constraints.ifbench_ratio import SentenceBalanceRatioIfbenchConstraint


def test_sentence_balance_all_types_evenly_distributed() -> None:
    value = "Calm day. Ready to go? What a surprise!"
    constraint = SentenceBalanceRatioIfbenchConstraint()

    success, reason = constraint.evaluate(value)

    assert success is True
    assert reason is None


def test_sentence_balance_fails_when_type_missing() -> None:
    value = "This is simple. That is plain. Nothing else here."
    constraint = SentenceBalanceRatioIfbenchConstraint()

    success, reason = constraint.evaluate(value)

    assert success is False
    assert reason is not None
    assert "Each sentence type must appear at least once" in reason


def test_sentence_balance_fails_when_distribution_is_skewed() -> None:
    value = "First statement. Second statement. Third statement. Is it fine? Amazing!"
    constraint = SentenceBalanceRatioIfbenchConstraint()

    success, reason = constraint.evaluate(value)

    assert success is False
    assert reason is not None
    assert "Sentence types are not balanced" in reason
