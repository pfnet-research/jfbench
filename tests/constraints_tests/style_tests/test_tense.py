import asyncio
import logging

from _pytest.logging import LogCaptureFixture
from conftest import StubLLMClient
import pytest

from jfbench.constraints.style import PastTenseStyleConstraint
from jfbench.constraints.style import PresentTenseStyleConstraint


PROMPT_HINT = "ユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。"


def test_past_tense_constraint_respects_llm_response(
    true_client: StubLLMClient,
) -> None:
    constraint = PastTenseStyleConstraint(true_client)
    assert asyncio.run(constraint.evaluate("I visited the museum yesterday."))[0]
    assert true_client.captured_prompt is not None
    assert "visited" in true_client.captured_prompt


def test_present_tense_constraint_rejects_when_llm_says_false(
    false_client: StubLLMClient,
) -> None:
    constraint = PresentTenseStyleConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("I had finished the work."))[0]


def test_past_tense_constraint_accepts_prefixed_true_response(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True: past tense confirmed"
    constraint = PastTenseStyleConstraint(true_client)
    assert asyncio.run(constraint.evaluate("She completed the review last night."))[0]


def test_past_tense_constraint_rejects_prefixed_false_response(
    false_client: StubLLMClient,
) -> None:
    false_client.reply = "False - not past tense"
    constraint = PastTenseStyleConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("She completes the review."))[0]


def test_present_tense_constraint_accepts_prefixed_true_response(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True: present tense detected"
    constraint = PresentTenseStyleConstraint(true_client)
    assert asyncio.run(constraint.evaluate("I deliver the update each week."))[0]


def test_present_tense_constraint_rejects_prefixed_false_response(
    false_client: StubLLMClient,
) -> None:
    false_client.reply = "False - tense mismatch"
    constraint = PresentTenseStyleConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("I delivered the update yesterday."))[0]


@pytest.mark.parametrize(
    ("constraint_cls", "text"),
    [
        (PastTenseStyleConstraint, "I deliver updates today."),
        (PresentTenseStyleConstraint, "I delivered updates yesterday."),
    ],
)
def test_tense_constraints_log_response_on_failure(
    constraint_cls: type[PastTenseStyleConstraint],
    text: str,
    false_client: StubLLMClient,
    caplog: LogCaptureFixture,
) -> None:
    response = f"False - {constraint_cls.__name__} tense issue"
    false_client.reply = response
    constraint = constraint_cls(false_client)
    with caplog.at_level(logging.INFO):
        assert not asyncio.run(constraint.evaluate(text))[0]
    assert response in caplog.text


@pytest.mark.parametrize(
    "constraint_cls",
    [
        PastTenseStyleConstraint,
        PresentTenseStyleConstraint,
    ],
)
def test_tense_style_prompts_focus_on_assistant_output_only(
    constraint_cls: type[PastTenseStyleConstraint],
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True"
    constraint = constraint_cls(true_client)
    asyncio.run(constraint.evaluate("dummy text"))
    assert true_client.captured_prompt is not None
    assert PROMPT_HINT in true_client.captured_prompt
