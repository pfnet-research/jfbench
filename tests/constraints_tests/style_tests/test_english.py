import asyncio
import logging

from _pytest.logging import LogCaptureFixture
from conftest import StubLLMClient
import pytest

from jfbench.constraints.style import AmericanEnglishStyleConstraint
from jfbench.constraints.style import BritishEnglishStyleConstraint
from jfbench.constraints.style import EnglishStyleConstraint
from jfbench.constraints.style import NoEnglishStyleConstraint


PROMPT_HINT = "ユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。"


def test_english_style_constraint_allows_english_text(true_client: StubLLMClient) -> None:
    true_client.reply = "True: English text confirmed"
    constraint = EnglishStyleConstraint(true_client)
    assert asyncio.run(constraint.evaluate("Hello world! This is a test."))[0]


def test_english_style_constraint_rejects_non_english_characters(
    false_client: StubLLMClient,
) -> None:
    false_client.reply = "False - mixture detected"
    constraint = EnglishStyleConstraint(false_client)
    result = asyncio.run(constraint.evaluate("Hello 世界"))
    assert result[0] is False
    assert result[1] is not None and "not recognized as English" in result[1]


def test_british_english_constraint_accepts_prefixed_true_response(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True: British spelling confirmed"
    constraint = BritishEnglishStyleConstraint(true_client)
    assert asyncio.run(constraint.evaluate("We appreciate your organisation."))[0]


def test_british_english_constraint_rejects_prefixed_false_response(
    false_client: StubLLMClient,
) -> None:
    false_client.reply = "False - not British English"
    constraint = BritishEnglishStyleConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("We appreciate your organization."))[0]


def test_american_english_constraint_accepts_prefixed_true_response(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True: American usage"
    constraint = AmericanEnglishStyleConstraint(true_client)
    assert asyncio.run(constraint.evaluate("Color and honor are standard spellings."))[0]


def test_american_english_constraint_rejects_prefixed_false_response(
    false_client: StubLLMClient,
) -> None:
    false_client.reply = "False - not American English"
    constraint = AmericanEnglishStyleConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("Colour and honour appear."))[0]


@pytest.mark.parametrize(
    ("constraint_cls", "text"),
    [
        (BritishEnglishStyleConstraint, "We appreciate your organization."),
        (AmericanEnglishStyleConstraint, "We appreciate your organisation."),
    ],
)
def test_variant_lm_constraints_log_response_on_failure(
    constraint_cls: type[BritishEnglishStyleConstraint],
    text: str,
    false_client: StubLLMClient,
    caplog: LogCaptureFixture,
) -> None:
    response = f"False - {constraint_cls.__name__} mismatch"
    false_client.reply = response
    constraint = constraint_cls(false_client)
    with caplog.at_level(logging.INFO):
        assert not asyncio.run(constraint.evaluate(text))[0]
    assert response in caplog.text


@pytest.mark.parametrize(
    "constraint_cls",
    [
        EnglishStyleConstraint,
        BritishEnglishStyleConstraint,
        AmericanEnglishStyleConstraint,
    ],
)
def test_english_style_prompts_focus_on_assistant_output_only(
    constraint_cls: type[EnglishStyleConstraint],
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True"
    constraint = constraint_cls(true_client)
    asyncio.run(constraint.evaluate("dummy text"))
    assert true_client.captured_prompt is not None
    assert PROMPT_HINT in true_client.captured_prompt


def test_no_english_constraint_rejects_when_llm_reports_false(
    false_client: StubLLMClient, caplog: LogCaptureFixture
) -> None:
    false_client.reply = "False - English letters found"
    constraint = NoEnglishStyleConstraint(false_client)
    with caplog.at_level(logging.INFO):
        result = asyncio.run(constraint.evaluate("Hello こんにちは"))
    assert result[0] is False
    assert "contains English elements" in str(result[1])
    assert "English letters found" in caplog.text


def test_no_english_constraint_accepts_true_response(true_client: StubLLMClient) -> None:
    true_client.reply = "True"
    constraint = NoEnglishStyleConstraint(true_client)
    assert asyncio.run(constraint.evaluate("こんにちは、世界"))[0]


def test_no_english_prompt_focuses_on_assistant_output_only(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True"
    constraint = NoEnglishStyleConstraint(true_client)
    asyncio.run(constraint.evaluate("dummy text"))
    assert true_client.captured_prompt is not None
    assert PROMPT_HINT in true_client.captured_prompt
