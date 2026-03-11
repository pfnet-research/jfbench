from jfbench.constraints.ifbench_sentence import AlliterationIncrementSentenceIfbenchConstraint


def test_alliteration_increment_accepts_increasing_runs() -> None:
    constraint = AlliterationIncrementSentenceIfbenchConstraint()

    text = "apple ant. banana boat ball. cat car cab cap."

    success, reason = constraint.evaluate(text)

    assert success is True
    assert reason is None


def test_alliteration_increment_uses_reading_and_normalization() -> None:
    constraint = AlliterationIncrementSentenceIfbenchConstraint()

    text = "ガラス ガレージ。バナナ パンダ バラ。"

    success, reason = constraint.evaluate(text)

    assert success is True
    assert reason is None


def test_alliteration_increment_rejects_non_increasing_runs() -> None:
    constraint = AlliterationIncrementSentenceIfbenchConstraint()

    text = "apple ant. berry bark."

    success, reason = constraint.evaluate(text)

    assert success is False
    assert reason is not None
