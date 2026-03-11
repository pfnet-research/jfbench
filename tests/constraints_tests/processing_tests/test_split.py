import pytest
from pytest import MonkeyPatch

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints.processing import SplitProcessingConstraint


def test_split_processing_constraint_returns_segmented_output() -> None:
    constraint = SplitProcessingConstraint("abcdef", 2)
    expected = "[\nabc,\ndef\n]"
    assert constraint.evaluate(f"Prefix\n{expected}\nSuffix")[0] is True
    assert constraint.evaluate("[\na,\nb,\nc,\nd,\ne,\nf\n]")[0] is False


def _capture_options(monkeypatch: MonkeyPatch) -> dict[str, list[str]]:
    captured: dict[str, list[str]] = {}

    def fake_random_instruction(self: ConstraintGroupMixin, options: list[str]) -> str:
        captured["options"] = list(options)
        return options[0]

    monkeypatch.setattr(ConstraintGroupMixin, "_random_instruction", fake_random_instruction)
    return captured


def test_split_instructions_templates(monkeypatch: MonkeyPatch) -> None:
    captured = _capture_options(monkeypatch)
    constraint = SplitProcessingConstraint("abcdef", 2, seed=0)

    instruction = constraint.instructions()
    options = captured["options"]

    assert instruction == options[0]
    assert len(options) == 5
    assert all("2" in option for option in options)
    assert all("<文字列1>" in option for option in options)
    assert all("含" in option for option in options)


def test_split_processing_constraint_rejects_non_positive_parts() -> None:
    with pytest.raises(ValueError):
        SplitProcessingConstraint("abcdef", 0)
