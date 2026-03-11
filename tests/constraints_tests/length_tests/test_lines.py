import pytest

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints.length.lines import BlankLinesLengthConstraint
from jfbench.constraints.length.lines import NewLinesLengthConstraint


def _capture_options(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    captured: dict[str, list[str]] = {}

    def fake_random_instruction(self: ConstraintGroupMixin, options: list[str]) -> str:
        captured["options"] = list(options)
        return options[0]

    monkeypatch.setattr(ConstraintGroupMixin, "_random_instruction", fake_random_instruction)
    return captured


def test_new_lines_length_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_options(monkeypatch)
    constraint = NewLinesLengthConstraint(2, seed=0)
    instruction = constraint.instructions()
    options = captured["options"]

    assert instruction == options[0]
    assert len(options) == 5
    assert all("2" in option for option in options)


def test_new_lines_length_instructions_test_variant() -> None:
    constraint = NewLinesLengthConstraint(3, seed=0)
    assert constraint.instructions(train_or_test="test") == "改行は合計で3回だけ入れてください。"


def test_blank_lines_length_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_options(monkeypatch)
    constraint = BlankLinesLengthConstraint(1, seed=0)
    instruction = constraint.instructions()
    options = captured["options"]

    assert instruction == options[0]
    assert len(options) == 5
    assert all("1" in option for option in options)


def test_blank_lines_length_instructions_test_variant() -> None:
    constraint = BlankLinesLengthConstraint(1, seed=0)
    assert constraint.instructions(train_or_test="test") == "空行は1行だけにしてください。"
