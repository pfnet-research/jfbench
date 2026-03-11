from jfbench.constraints.ifbench_sentence import IncrementSentenceIfbenchConstraint


def test_increment_sentence_accepts_exact_step() -> None:
    constraint = IncrementSentenceIfbenchConstraint(increment=1)

    text = "a b. a b c"

    success, reason = constraint.evaluate(text)

    assert success is True
    assert reason is None


def test_increment_sentence_rejects_wrong_step() -> None:
    constraint = IncrementSentenceIfbenchConstraint(increment=2)

    text = "a b. a b c"

    success, reason = constraint.evaluate(text)

    assert success is False
    assert reason is not None


def test_increment_sentence_requires_multiple_sentences() -> None:
    constraint = IncrementSentenceIfbenchConstraint(increment=1)

    success, reason = constraint.evaluate("only one sentence")

    assert success is False
    assert reason is not None
