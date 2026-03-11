import asyncio
import logging

from _pytest.logging import LogCaptureFixture
from conftest import StubLLMClient
import pytest

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints.style import FirstPersonPluralStyleConstraint
from jfbench.constraints.style import FirstPersonSingularStyleConstraint
from jfbench.constraints.style import ImpersonalStyleConstraint
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
    ("constraint_cls", "reason_snippet", "text"),
    [
        (FirstPersonSingularStyleConstraint, "lacks first-person singular", "これは例文です。"),
        (FirstPersonPluralStyleConstraint, "lacks first-person plural", "これは例文です。"),
        (ImpersonalStyleConstraint, "contains first-person references", "私は客観的に述べる。"),
    ],
)
def test_first_person_constraints_log_false_response(
    constraint_cls: type[FirstPersonSingularStyleConstraint],
    reason_snippet: str,
    text: str,
    false_client: StubLLMClient,
    caplog: LogCaptureFixture,
) -> None:
    false_client.reply = "False - missing requirement"
    constraint = constraint_cls(false_client)
    with caplog.at_level(logging.INFO):
        result = asyncio.run(constraint.evaluate(text))
    assert result[0] is False
    assert reason_snippet in str(result[1])
    assert "missing requirement" in caplog.text


@pytest.mark.parametrize(
    ("constraint_cls", "text"),
    [
        (FirstPersonSingularStyleConstraint, "私はここにいる。"),
        (FirstPersonPluralStyleConstraint, "われわれはここにいる。"),
        (ImpersonalStyleConstraint, "手順に従って処理を行う。"),
    ],
)
def test_first_person_constraints_accept_true_response(
    constraint_cls: type[FirstPersonSingularStyleConstraint],
    text: str,
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True"
    constraint = constraint_cls(true_client)
    assert asyncio.run(constraint.evaluate(text))[0]


@pytest.mark.parametrize(
    "constraint_cls",
    [
        FirstPersonSingularStyleConstraint,
        FirstPersonPluralStyleConstraint,
        ImpersonalStyleConstraint,
    ],
)
def test_first_person_prompts_focus_on_assistant_output_only(
    constraint_cls: type[FirstPersonSingularStyleConstraint],
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
        (FirstPersonSingularStyleConstraint(StubLLMClient("True"), seed=0), "私"),
        (FirstPersonPluralStyleConstraint(StubLLMClient("True"), seed=0), "われわれ"),
        (ImpersonalStyleConstraint(StubLLMClient("True"), seed=0), "一人称"),
    ],
)
def test_first_person_instruction_templates(
    monkeypatch: pytest.MonkeyPatch, constraint: Constraint, expected_keyword: str
) -> None:
    captured = _capture_options(monkeypatch)
    instruction = constraint.instructions()
    options = captured["options"]

    assert instruction == options[0]
    assert len(options) == 5
    assert expected_keyword in instruction
