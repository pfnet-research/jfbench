import asyncio
import logging

from _pytest.logging import LogCaptureFixture
from conftest import StubLLMClient
import pytest

from jfbench.constraints.style import AcademicToneStyleConstraint
from jfbench.constraints.style import BusinessToneStyleConstraint
from jfbench.constraints.style import CasualToneStyleConstraint
from jfbench.constraints.style import FormalToneStyleConstraint


PROMPT_HINT = "ユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。"


def test_formal_tone_constraint_accepts_formal_text(
    true_client: StubLLMClient,
) -> None:
    constraint = FormalToneStyleConstraint(true_client)
    assert asyncio.run(constraint.evaluate("We sincerely appreciate your continued support."))[0]
    assert true_client.captured_prompt is not None
    assert "support" in true_client.captured_prompt


def test_casual_tone_constraint_rejects_when_llm_says_false(
    false_client: StubLLMClient,
) -> None:
    constraint = CasualToneStyleConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("Ok, cool."))[0]


@pytest.mark.parametrize(
    ("constraint_factory", "sample_text"),
    [
        (FormalToneStyleConstraint, "We sincerely appreciate your patience."),
        (CasualToneStyleConstraint, "Hey, thanks for the heads-up!"),
        (AcademicToneStyleConstraint, "The study indicates a significant variance."),
        (BusinessToneStyleConstraint, "Please proceed once the contract is signed."),
    ],
)
def test_tone_constraints_accept_prefixed_true_response(
    true_client: StubLLMClient,
    constraint_factory: type,
    sample_text: str,
) -> None:
    true_client.reply = "True: tone verified"
    constraint = constraint_factory(true_client)
    assert asyncio.run(constraint.evaluate(sample_text))[0]


@pytest.mark.parametrize(
    "constraint_factory",
    [
        FormalToneStyleConstraint,
        CasualToneStyleConstraint,
        AcademicToneStyleConstraint,
        BusinessToneStyleConstraint,
    ],
)
def test_tone_constraints_reject_prefixed_false_response(
    false_client: StubLLMClient,
    constraint_factory: type,
    caplog: LogCaptureFixture,
) -> None:
    false_client.reply = "False - tone mismatch"
    constraint = constraint_factory(false_client)
    with caplog.at_level(logging.INFO):
        assert not asyncio.run(constraint.evaluate("Placeholder text."))[0]
    assert "False - tone mismatch" in caplog.text


@pytest.mark.parametrize(
    "constraint_cls",
    [
        FormalToneStyleConstraint,
        CasualToneStyleConstraint,
        AcademicToneStyleConstraint,
        BusinessToneStyleConstraint,
    ],
)
def test_tone_style_prompts_focus_on_assistant_output_only(
    constraint_cls: type[FormalToneStyleConstraint],
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True"
    constraint = constraint_cls(true_client)
    asyncio.run(constraint.evaluate("dummy text"))
    assert true_client.captured_prompt is not None
    assert PROMPT_HINT in true_client.captured_prompt
