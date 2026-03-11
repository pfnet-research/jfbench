from pytest import MonkeyPatch

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints.processing import ReplacementProcessingConstraint


def test_replacement_processing_constraint_validates_replaced_text() -> None:
    constraint = ReplacementProcessingConstraint("abcdef", 1, 3, "hoge")
    assert constraint.evaluate("prefix ahogeef suffix")[0] is True
    assert constraint.evaluate("abcdef")[0] is False


def _capture_options(monkeypatch: MonkeyPatch) -> dict[str, list[str]]:
    captured: dict[str, list[str]] = {}

    def fake_random_instruction(self: ConstraintGroupMixin, options: list[str]) -> str:
        captured["options"] = list(options)
        return options[0]

    monkeypatch.setattr(ConstraintGroupMixin, "_random_instruction", fake_random_instruction)
    return captured


def test_replacement_instructions_templates(monkeypatch: MonkeyPatch) -> None:
    captured = _capture_options(monkeypatch)
    constraint = ReplacementProcessingConstraint("abcdef", 1, 3, "hoge", seed=0)

    instruction = constraint.instructions()
    options = captured["options"]

    assert instruction == options[0]
    assert len(options) == 5
    assert all(token in option for option in options for token in ("1", "3", "hoge"))
