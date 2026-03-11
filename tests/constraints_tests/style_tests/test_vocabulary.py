import asyncio
import logging

from _pytest.logging import LogCaptureFixture
from conftest import StubLLMClient
import pytest

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints.style import DifficultVocacularyStyleConstraint
from jfbench.constraints.style import EasyVocabularyStyleConstraint
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
        (EasyVocabularyStyleConstraint, "is not child-friendly"),
        (DifficultVocacularyStyleConstraint, "is too simple"),
    ],
)
def test_vocabulary_constraints_log_false_response(
    constraint_cls: type[EasyVocabularyStyleConstraint],
    reason_snippet: str,
    false_client: StubLLMClient,
    caplog: LogCaptureFixture,
) -> None:
    false_client.reply = "False - vocabulary mismatch"
    constraint = constraint_cls(false_client)
    with caplog.at_level(logging.INFO):
        result = asyncio.run(constraint.evaluate("sample text"))
    assert result[0] is False
    assert reason_snippet in str(result[1])
    assert "vocabulary mismatch" in caplog.text


@pytest.mark.parametrize(
    "constraint_cls",
    [
        EasyVocabularyStyleConstraint,
        DifficultVocacularyStyleConstraint,
    ],
)
def test_vocabulary_constraints_accept_true_response(
    constraint_cls: type[EasyVocabularyStyleConstraint],
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True"
    constraint = constraint_cls(true_client)
    assert asyncio.run(constraint.evaluate("text"))[0]


@pytest.mark.parametrize(
    "constraint_cls",
    [
        EasyVocabularyStyleConstraint,
        DifficultVocacularyStyleConstraint,
    ],
)
def test_vocabulary_prompts_focus_on_assistant_output_only(
    constraint_cls: type[EasyVocabularyStyleConstraint],
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
        (EasyVocabularyStyleConstraint(StubLLMClient("True"), seed=0), "やさしい"),
        (DifficultVocacularyStyleConstraint(StubLLMClient("True"), seed=0), "専門"),
    ],
)
def test_vocabulary_instruction_templates(
    monkeypatch: pytest.MonkeyPatch, constraint: Constraint, expected_keyword: str
) -> None:
    captured = _capture_options(monkeypatch)
    instruction = constraint.instructions()
    options = captured["options"]

    assert instruction == options[0]
    assert len(options) == 5
    assert expected_keyword in instruction
