from jfbench.constraints.ifbench_count import UniqueWordCountIfbenchConstraint


def test_unique_words_counts_japanese_tokens() -> None:
    constraint = UniqueWordCountIfbenchConstraint(minimum_unique=6)
    value = "リンゴを食べ、バナナを食べ、みかんを食べた。"

    success, reason = constraint.evaluate(value)

    assert success is True
    assert reason is None


def test_unique_words_detects_shortfall() -> None:
    constraint = UniqueWordCountIfbenchConstraint(minimum_unique=10)
    value = "リンゴを食べ、バナナを食べ、みかんを食べた。"

    success, reason = constraint.evaluate(value)

    assert success is False
    assert reason is not None
    assert "Not enough unique words" in reason
