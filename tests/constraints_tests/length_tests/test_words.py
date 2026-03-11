import pytest

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints.length.words import WordsLengthConstraint


def _capture_options(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    captured: dict[str, list[str]] = {}

    def fake_random_instruction(self: ConstraintGroupMixin, options: list[str]) -> str:
        captured["options"] = list(options)
        return options[0]

    monkeypatch.setattr(ConstraintGroupMixin, "_random_instruction", fake_random_instruction)
    return captured


def test_word_length_instructions_range(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_options(monkeypatch)
    constraint = WordsLengthConstraint(3, 5, seed=0)
    instruction = constraint.instructions()
    options = captured["options"]

    assert instruction == options[0]
    assert len(options) == 5
    assert all("3" in option and "5" in option for option in options)


def test_word_length_instructions_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    constraint = WordsLengthConstraint(3, 5, seed=0)

    train_instruction = constraint.instructions()
    test_instruction = constraint.instructions(train_or_test="test")

    assert test_instruction == "語数は3語以上5語以下に揃えてください。"
    assert test_instruction != train_instruction


def test_word_length_constraint_counts_japanese_tokens() -> None:
    constraint = WordsLengthConstraint(3, 6)

    ok_result = constraint.evaluate("猫が好きです。")
    assert ok_result[0] is True
    assert ok_result[1] is None

    too_few = constraint.evaluate("猫。")
    assert too_few[0] is False
    assert "[Word Length] Too few words (2); minimum is 3." == too_few[1]

    too_many = constraint.evaluate("猫が好きです。犬も好きです。")
    assert too_many[0] is False
    assert "[Word Length] Too many words (10); maximum is 6." == too_many[1]
