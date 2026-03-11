from jfbench.constraints.character.punctuation import JapanesePunctuationConstraint
from jfbench.constraints.character.punctuation import NoCommasConstraint
from jfbench.constraints.character.punctuation import NoJapanesePunctuationConstraint


def test_japanese_punctuation_allows_decimal_point_after_digit() -> None:
    constraint = JapanesePunctuationConstraint()

    result, reason = constraint.evaluate("価格は3.14です")

    assert result is True
    assert reason is None


def test_japanese_punctuation_rejects_non_digit_punctuation() -> None:
    constraint = JapanesePunctuationConstraint()

    result, reason = constraint.evaluate("a. b")

    assert result is False
    assert reason is not None


def test_no_japanese_punctuation_allows_after_digit() -> None:
    constraint = NoJapanesePunctuationConstraint()

    result, reason = constraint.evaluate("1、2")

    assert result is True
    assert reason is None


def test_no_japanese_punctuation_rejects_non_digit_punctuation() -> None:
    constraint = NoJapanesePunctuationConstraint()

    result, reason = constraint.evaluate("a、b")

    assert result is False
    assert reason is not None


def test_no_commas_allows_after_digit() -> None:
    constraint = NoCommasConstraint()

    result, reason = constraint.evaluate("1,2")

    assert result is True
    assert reason is None


def test_no_commas_rejects_non_digit_punctuation() -> None:
    constraint = NoCommasConstraint()

    result, reason = constraint.evaluate("a, b")

    assert result is False
    assert reason is not None
