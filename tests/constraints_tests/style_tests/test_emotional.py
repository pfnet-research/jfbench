import asyncio
import logging

from _pytest.logging import LogCaptureFixture
from conftest import StubLLMClient
import pytest

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints.style import AngryEmotionalStyleConstraint
from jfbench.constraints.style import HappyEmotionalStyleConstraint
from jfbench.constraints.style import JoyfulEmotionalStyleConstraint
from jfbench.constraints.style import SadEmotionalStyleConstraint
from jfbench.protocol import Constraint


PROMPT_HINT = "ユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。"


def _capture_options(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    captured: dict[str, list[str]] = {}

    def fake_random_instruction(self: ConstraintGroupMixin, options: list[str]) -> str:
        captured["options"] = list(options)
        return options[0]

    monkeypatch.setattr(ConstraintGroupMixin, "_random_instruction", fake_random_instruction)
    return captured


@pytest.mark.parametrize(
    ("constraint_cls", "reason_snippet"),
    [
        (JoyfulEmotionalStyleConstraint, "lacks a joyful tone"),
        (AngryEmotionalStyleConstraint, "lacks an angry tone"),
        (SadEmotionalStyleConstraint, "lacks a sad tone"),
        (HappyEmotionalStyleConstraint, "lacks a happy tone"),
    ],
)
def test_emotional_constraints_log_false_response(
    constraint_cls: type[JoyfulEmotionalStyleConstraint],
    reason_snippet: str,
    false_client: StubLLMClient,
    caplog: LogCaptureFixture,
) -> None:
    false_client.reply = "False - missing emotion"
    constraint = constraint_cls(false_client)
    with caplog.at_level(logging.INFO):
        result = asyncio.run(constraint.evaluate("example text"))
    assert result[0] is False
    assert reason_snippet in str(result[1])
    assert "missing emotion" in caplog.text


@pytest.mark.parametrize(
    "constraint_cls",
    [
        JoyfulEmotionalStyleConstraint,
        AngryEmotionalStyleConstraint,
        SadEmotionalStyleConstraint,
        HappyEmotionalStyleConstraint,
    ],
)
def test_emotional_constraints_accept_true_response(
    constraint_cls: type[JoyfulEmotionalStyleConstraint],
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True"
    constraint = constraint_cls(true_client)
    assert asyncio.run(constraint.evaluate("text"))[0]


@pytest.mark.parametrize(
    "constraint_cls",
    [
        JoyfulEmotionalStyleConstraint,
        AngryEmotionalStyleConstraint,
        SadEmotionalStyleConstraint,
        HappyEmotionalStyleConstraint,
    ],
)
def test_emotional_prompts_focus_on_assistant_output_only(
    constraint_cls: type[JoyfulEmotionalStyleConstraint],
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True"
    constraint = constraint_cls(true_client)
    asyncio.run(constraint.evaluate("dummy text"))
    assert true_client.captured_prompt is not None
    assert PROMPT_HINT in true_client.captured_prompt


@pytest.mark.parametrize(
    ("constraint", "expected_keyword"),
    [
        (JoyfulEmotionalStyleConstraint(StubLLMClient("True"), seed=0), "嬉"),
        (AngryEmotionalStyleConstraint(StubLLMClient("True"), seed=0), "怒"),
        (SadEmotionalStyleConstraint(StubLLMClient("True"), seed=0), "悲"),
        (HappyEmotionalStyleConstraint(StubLLMClient("True"), seed=0), "幸せ"),
    ],
)
def test_emotional_instruction_templates(
    monkeypatch: pytest.MonkeyPatch, constraint: Constraint, expected_keyword: str
) -> None:
    captured = _capture_options(monkeypatch)
    instruction = constraint.instructions()
    options = captured["options"]

    assert instruction == options[0]
    assert len(options) == 5
    assert expected_keyword in instruction
