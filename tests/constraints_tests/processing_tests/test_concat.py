from pytest import MonkeyPatch

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints.processing import ConcatProcessingConstraint


def test_concat_processing_constraint_requires_repeated_document() -> None:
    constraint = ConcatProcessingConstraint("abc", 2)
    assert constraint.evaluate("abcabc")[0] is True
    assert constraint.evaluate("abc")[0] is False


def _capture_options(monkeypatch: MonkeyPatch) -> dict[str, list[str]]:
    captured: dict[str, list[str]] = {}

    def fake_random_instruction(self: ConstraintGroupMixin, options: list[str]) -> str:
        captured["options"] = list(options)
        return options[0]

    monkeypatch.setattr(ConstraintGroupMixin, "_random_instruction", fake_random_instruction)
    return captured


def test_concat_instructions_templates(monkeypatch: MonkeyPatch) -> None:
    captured = _capture_options(monkeypatch)
    constraint = ConcatProcessingConstraint("abc", 2, seed=0)

    instruction = constraint.instructions()
    options = captured["options"]

    assert instruction.startswith(options[0])
    assert instruction.endswith("区切り文字は含めないでください。")
    assert len(options) == 5
    assert all("2" in option for option in options)
