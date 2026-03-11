from jfbench.constraints.ifbench_ratio import SentenceTypeRatioIfbenchConstraint


def test_sentence_type_ratio_passes_near_two_to_one() -> None:
    value = "Statement one. Statement two. Curious?"
    constraint = SentenceTypeRatioIfbenchConstraint()

    success, reason = constraint.evaluate(value)

    assert success is True
    assert reason is None


def test_sentence_type_ratio_requires_question() -> None:
    value = "Only statements here. Another one."
    constraint = SentenceTypeRatioIfbenchConstraint()

    success, reason = constraint.evaluate(value)

    assert success is False
    assert reason is not None
    assert "At least one question is required" in reason


def test_sentence_type_ratio_rejects_skewed_distribution() -> None:
    value = "One. Two. Three. Are you okay?"
    constraint = SentenceTypeRatioIfbenchConstraint()

    success, reason = constraint.evaluate(value)

    assert success is False
    assert reason is not None
    assert "not close to 2:1" in reason
