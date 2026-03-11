import asyncio
import logging

from _pytest.logging import LogCaptureFixture
from conftest import StubLLMClient
import pytest

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints.style import PlainStyleConstraint
from jfbench.protocol import Constraint


PROMPT_HINT = "ユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。"


def _capture_options(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    captured: dict[str, list[str]] = {}

    def fake_random_instruction(self: ConstraintGroupMixin, options: list[str]) -> str:
        captured["options"] = list(options)
        return options[0]

    monkeypatch.setattr(ConstraintGroupMixin, "_random_instruction", fake_random_instruction)
    return captured


def test_plain_style_constraint_rejects_polite_form(
    false_client: StubLLMClient, caplog: LogCaptureFixture
) -> None:
    false_client.reply = "False - polite tone found"
    constraint = PlainStyleConstraint(false_client)
    with caplog.at_level(logging.INFO):
        result = asyncio.run(constraint.evaluate("今日は静かです。"))
    assert result[0] is False
    assert "not in plain form" in str(result[1])
    assert "polite tone found" in caplog.text


def test_plain_style_constraint_accepts_plain_form(true_client: StubLLMClient) -> None:
    true_client.reply = "True"
    constraint = PlainStyleConstraint(true_client)
    assert asyncio.run(constraint.evaluate("今日は静かだ。"))[0]


def test_plain_style_prompt_focuses_on_assistant_output_only(true_client: StubLLMClient) -> None:
    true_client.reply = "True"
    constraint = PlainStyleConstraint(true_client)
    asyncio.run(constraint.evaluate("dummy text"))
    assert true_client.captured_prompt is not None
    assert PROMPT_HINT in true_client.captured_prompt


@pytest.mark.parametrize(
    "constraint",
    [
        PlainStyleConstraint(StubLLMClient("True"), seed=0),
    ],
)
def test_plain_style_instruction_templates(
    monkeypatch: pytest.MonkeyPatch, constraint: Constraint
) -> None:
    captured = _capture_options(monkeypatch)
    instruction = constraint.instructions()
    options = captured["options"]

    assert instruction == options[0]
    assert len(options) == 5
    assert "常体" in instruction
