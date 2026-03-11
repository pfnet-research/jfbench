import asyncio
import logging

from _pytest.logging import LogCaptureFixture
from conftest import StubLLMClient
import pytest
from pytest import MonkeyPatch

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints.processing import ExtractionProcessingConstraint
from jfbench.constraints.processing import PrefixExtractionProcessingConstraint
from jfbench.constraints.processing import RangeExtractionProcessingConstraint
from jfbench.constraints.processing import SuffixExtractionProcessingConstraint


def test_extraction_constraint_confirms_value_with_llm(
    true_client: StubLLMClient,
) -> None:
    document = "Alpha Beta Gamma"
    constraint = ExtractionProcessingConstraint(true_client, document, "contains Beta")

    assert asyncio.run(constraint.evaluate("Beta"))[0]
    assert true_client.captured_prompt is not None
    assert "Beta" in true_client.captured_prompt
    assert "Alpha Beta Gamma" in true_client.captured_prompt


def test_extraction_constraint_accepts_prefixed_true_response(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True: extraction correct"
    constraint = ExtractionProcessingConstraint(true_client, "Doc", "Detail")
    assert asyncio.run(constraint.evaluate("Detail value"))[0]


def test_extraction_constraint_rejects_prefixed_false_response(
    false_client: StubLLMClient,
) -> None:
    false_client.reply = "False - extraction mismatch"
    constraint = ExtractionProcessingConstraint(false_client, "Doc", "Detail")
    assert not asyncio.run(constraint.evaluate("Wrong value"))[0]


def test_extraction_constraint_logs_llm_response_on_failure(
    false_client: StubLLMClient,
    caplog: LogCaptureFixture,
) -> None:
    false_client.reply = "False - extracted value invalid"
    constraint = ExtractionProcessingConstraint(false_client, "Doc", "Detail")
    with caplog.at_level(logging.INFO):
        assert not asyncio.run(constraint.evaluate("Invalid snippet"))[0]
    assert "False - extracted value invalid" in caplog.text


def test_prefix_extraction_processing_constraint_requires_prefix_in_output() -> None:
    document = "abcdefg"
    constraint = PrefixExtractionProcessingConstraint(document, 3)
    assert constraint.evaluate("abc is present here")[0] is True
    assert constraint.evaluate("no prefix here")[0] is False


def test_suffix_and_range_extraction_processing_constraint_require_fragments() -> None:
    suffix_constraint = SuffixExtractionProcessingConstraint("abcdef", 3)
    range_constraint = RangeExtractionProcessingConstraint("abcdef", 1, 4)

    assert suffix_constraint.evaluate("zzzdef")[0] is True
    assert suffix_constraint.evaluate("zzzde")[0] is False

    assert range_constraint.evaluate("xbcdez")[0] is True
    assert range_constraint.evaluate("xbcdz")[0] is False


def _capture_options(monkeypatch: MonkeyPatch) -> dict[str, list[str]]:
    captured: dict[str, list[str]] = {}

    def fake_random_instruction(self: ConstraintGroupMixin, options: list[str]) -> str:
        captured["options"] = list(options)
        return options[0]

    monkeypatch.setattr(ConstraintGroupMixin, "_random_instruction", fake_random_instruction)
    return captured


@pytest.mark.parametrize(
    ("constraint", "expected_tokens"),
    [
        (PrefixExtractionProcessingConstraint("abcdef", 2, seed=0), ("2",)),
        (SuffixExtractionProcessingConstraint("abcdef", 3, seed=0), ("3",)),
        (RangeExtractionProcessingConstraint("abcdef", 1, 4, seed=0), ("1", "4")),
    ],
)
def test_extraction_instructions_templates(
    monkeypatch: MonkeyPatch,
    constraint: PrefixExtractionProcessingConstraint
    | SuffixExtractionProcessingConstraint
    | RangeExtractionProcessingConstraint,
    expected_tokens: tuple[str, ...],
) -> None:
    captured = _capture_options(monkeypatch)
    instruction = constraint.instructions()
    options = captured["options"]

    assert instruction.startswith(options[0])
    if isinstance(constraint, RangeExtractionProcessingConstraint):
        assert "0文字目" in instruction
    assert len(options) == 5
    for token in expected_tokens:
        assert all(token in option for option in options)
