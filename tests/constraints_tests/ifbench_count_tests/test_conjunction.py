import pytest

from jfbench.constraints.ifbench_count import ConjunctionCountIfbenchConstraint


def test_conjunction_counts_japanese_terms() -> None:
    constraint = ConjunctionCountIfbenchConstraint(minimum_kinds=3)
    value = "今日は晴れて、そして散歩した。しかし休憩し、さらに本を読んだ。"

    success, reason = constraint.evaluate(value)

    assert success is True
    assert reason is None


def test_conjunction_counts_when_embedded_in_text() -> None:
    constraint = ConjunctionCountIfbenchConstraint(minimum_kinds=2)
    value = "だからこそ続けるし、つまり結論は明らかだ。"

    success, reason = constraint.evaluate(value)

    assert success is True
    assert reason is None


def test_conjunction_fails_with_insufficient_conjunctions() -> None:
    constraint = ConjunctionCountIfbenchConstraint(minimum_kinds=2)
    value = "接続詞が足りない文です。"

    success, reason = constraint.evaluate(value)

    assert success is False
    assert reason is not None
    assert "Not enough distinct conjunctions" in reason


def test_conjunction_group_inferred_from_path() -> None:
    constraint = ConjunctionCountIfbenchConstraint(minimum_kinds=1)

    assert constraint.group == "IfbenchCount"


def test_conjunction_instructions_reject_train_mode() -> None:
    constraint = ConjunctionCountIfbenchConstraint(minimum_kinds=1)
    with pytest.raises(ValueError, match="ifbench"):
        constraint.instructions(train_or_test="train")
