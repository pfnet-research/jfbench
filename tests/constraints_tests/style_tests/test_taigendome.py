import asyncio
import logging

from _pytest.logging import LogCaptureFixture
from conftest import StubLLMClient
import pytest

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints.style import TaigendomeStyleConstraint
from jfbench.protocol import Constraint


PROMPT_HINT = "ユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。"


def _capture_options(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    captured: dict[str, list[str]] = {}

    def fake_random_instruction(self: ConstraintGroupMixin, options: list[str]) -> str:
        captured["options"] = list(options)
        return options[0]

    monkeypatch.setattr(ConstraintGroupMixin, "_random_instruction", fake_random_instruction)
    return captured


def test_taigendome_constraint_rejects_non_nominal_endings(
    false_client: StubLLMClient, caplog: LogCaptureFixture
) -> None:
    false_client.reply = "False - verb ending detected"
    constraint = TaigendomeStyleConstraint(false_client)
    with caplog.at_level(logging.INFO):
        result = asyncio.run(constraint.evaluate("この文章は体言止めではない。"))
    assert result[0] is False
    assert "does not end sentences with nouns" in str(result[1])
    assert "verb ending detected" in caplog.text


def test_taigendome_constraint_accepts_nominal_endings(true_client: StubLLMClient) -> None:
    true_client.reply = "True"
    constraint = TaigendomeStyleConstraint(true_client)
    assert asyncio.run(constraint.evaluate("結論、積極的な採用。"))[0]


def test_taigendome_prompt_focuses_on_assistant_output_only(true_client: StubLLMClient) -> None:
    true_client.reply = "True"
    constraint = TaigendomeStyleConstraint(true_client)
    asyncio.run(constraint.evaluate("dummy text"))
    assert true_client.captured_prompt is not None
    assert PROMPT_HINT in true_client.captured_prompt


@pytest.mark.parametrize(
    "constraint",
    [
        TaigendomeStyleConstraint(StubLLMClient("True"), seed=0),
    ],
)
def test_taigendome_instruction_templates(
    monkeypatch: pytest.MonkeyPatch, constraint: Constraint
) -> None:
    captured = _capture_options(monkeypatch)
    instruction = constraint.instructions()
    options = captured["options"]

    assert instruction == options[0]
    assert len(options) == 5
    assert "体言" in instruction
