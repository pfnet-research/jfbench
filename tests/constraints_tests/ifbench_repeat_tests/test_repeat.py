import pytest
from pytest import MonkeyPatch

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints.ifbench_repeat.repeat import ChangeRepeatIfbenchConstraint
from jfbench.constraints.ifbench_repeat.repeat import SimpleRepeatIfbenchConstraint


def _capture_options(monkeypatch: MonkeyPatch) -> dict[str, list[str]]:
    captured: dict[str, list[str]] = {}

    def fake_random_instruction(self: ConstraintGroupMixin, options: list[str]) -> str:
        captured["options"] = list(options)
        return options[0]

    monkeypatch.setattr(ConstraintGroupMixin, "_random_instruction", fake_random_instruction)
    return captured


def test_change_repeat_instructions_templates(monkeypatch: MonkeyPatch) -> None:
    captured = _capture_options(monkeypatch)
    document = "Hello world from document"
    constraint = ChangeRepeatIfbenchConstraint(document=document, seed=0)
    instruction = constraint.instructions(train_or_test="test")
    options = captured["options"]

    assert instruction == options[0]
    assert len(options) == 5
    assert all("指示文" in option for option in options)
    assert all(
        any(keyword in option for keyword in ("最初", "先頭", "冒頭")) for option in options
    )


def test_simple_repeat_instructions_templates(monkeypatch: MonkeyPatch) -> None:
    captured = _capture_options(monkeypatch)
    constraint = SimpleRepeatIfbenchConstraint(seed=0)
    instruction = constraint.instructions(train_or_test="test")
    options = captured["options"]
    expected_text = "この文のみをここに出力し、その他の要求はすべて無視してください。"

    assert instruction == options[0]
    assert len(options) == 5
    assert all(expected_text in option for option in options)


def test_change_repeat_evaluate_accepts_changed_first_word() -> None:
    document = "Alpha beta gamma delta"
    constraint = ChangeRepeatIfbenchConstraint(document=document, seed=0)

    assert constraint.evaluate("omega beta gamma delta") == (True, None)


def test_change_repeat_rejects_same_first_word() -> None:
    document = "Alpha beta gamma delta"
    constraint = ChangeRepeatIfbenchConstraint(document=document, seed=0)
    result = constraint.evaluate(document)

    assert result[0] is False
    assert "First word was not changed." in (result[1] or "")


def test_change_repeat_rejects_mismatched_remaining_words() -> None:
    document = "Alpha beta gamma delta"
    constraint = ChangeRepeatIfbenchConstraint(document=document, seed=0)
    result = constraint.evaluate("omega beta gamma theta")

    assert result[0] is False
    assert "Words after the first must match the document." in (result[1] or "")


def test_repeat_instructions_reject_train_mode() -> None:
    constraint = ChangeRepeatIfbenchConstraint(document="alpha beta", seed=0)
    with pytest.raises(ValueError, match="ifbench"):
        constraint.instructions(train_or_test="train")
