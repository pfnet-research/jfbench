import pytest

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints.format.sentence_delimiter import SentenceDelimiterFormatConstraint


def test_sentence_delimiter_evaluate_valid_when_pysbd_matches() -> None:
    constraint = SentenceDelimiterFormatConstraint("|")
    assert constraint.evaluate("これは文|あれも文|")[0] is True


def test_sentence_delimiter_evaluate_invalid_when_pysbd_splits_more() -> None:
    constraint = SentenceDelimiterFormatConstraint("|")
    assert constraint.evaluate("これは文。追加の文|それも文|")[0] is False


def test_sentence_delimiter_evaluate_invalid_without_segments() -> None:
    constraint = SentenceDelimiterFormatConstraint("。")
    assert constraint.evaluate("。。")[0] is False


def test_sentence_delimiter_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_random_instruction(self: ConstraintGroupMixin, options: list[str]) -> str:
        captured["options"] = list(options)
        return options[0]

    monkeypatch.setattr(ConstraintGroupMixin, "_random_instruction", fake_random_instruction)
    constraint = SentenceDelimiterFormatConstraint("。", seed=0)
    instruction = constraint.instructions()

    assert instruction == captured["options"][0]
    assert len(captured["options"]) == 5
    assert all("。" in option for option in captured["options"])
