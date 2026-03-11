import asyncio
import logging

from _pytest.logging import LogCaptureFixture
from conftest import StubLLMClient

from jfbench.constraints.meta_output import GreetingOutputConstraint
from jfbench.constraints.meta_output import NoGreetingOutputConstraint


def test_greeting_output_constraint_accepts_when_llm_confirms(true_client: StubLLMClient) -> None:
    constraint = GreetingOutputConstraint(true_client)
    assert asyncio.run(constraint.evaluate("Hello team,\nHere is the report."))[0]
    assert true_client.captured_prompt is not None
    assert "Hello team" in true_client.captured_prompt


def test_greeting_output_constraint_rejects_when_llm_disagrees(
    false_client: StubLLMClient,
) -> None:
    constraint = GreetingOutputConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("Summary follows."))[0]


def test_greeting_output_constraint_logs_llm_response_on_failure(
    false_client: StubLLMClient, caplog: LogCaptureFixture
) -> None:
    false_client.reply = "False - greeting absent"
    constraint = GreetingOutputConstraint(false_client)
    with caplog.at_level(logging.INFO):
        assert not asyncio.run(constraint.evaluate("Report begins now."))[0]
    assert "greeting absent" in caplog.text


def test_greeting_output_constraint_instructions_test_variant(
    true_client: StubLLMClient,
) -> None:
    constraint = GreetingOutputConstraint(true_client, seed=0)
    assert constraint.instructions(train_or_test="test") == "冒頭に短い挨拶を一つ置いてください。"


def test_no_greeting_constraint_accepts_when_llm_confirms(true_client: StubLLMClient) -> None:
    constraint = NoGreetingOutputConstraint(true_client)
    assert asyncio.run(constraint.evaluate("Report:\n- Item 1\n- Item 2"))[0]
    assert true_client.captured_prompt is not None
    assert "Report" in true_client.captured_prompt


def test_no_greeting_constraint_rejects_when_llm_disagrees(false_client: StubLLMClient) -> None:
    constraint = NoGreetingOutputConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("Hello there, here is the report."))[0]


def test_no_greeting_constraint_accepts_prefixed_true_response(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True: starts directly"
    constraint = NoGreetingOutputConstraint(true_client)
    assert asyncio.run(constraint.evaluate("Agenda\n1. Budget\n2. Timeline"))[0]


def test_no_greeting_constraint_rejects_prefixed_false_response(
    false_client: StubLLMClient,
) -> None:
    false_client.reply = "False - greeting detected"
    constraint = NoGreetingOutputConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("Hi team, summary below."))[0]


def test_no_greeting_constraint_logs_llm_response_on_failure(
    false_client: StubLLMClient,
    caplog: LogCaptureFixture,
) -> None:
    false_client.reply = "False - intro greeting detected"
    constraint = NoGreetingOutputConstraint(false_client)
    with caplog.at_level(logging.INFO):
        assert not asyncio.run(constraint.evaluate("Hello everyone, agenda follows."))[0]
    assert "False - intro greeting detected" in caplog.text


def test_no_greeting_output_constraint_instructions_test_variant(
    true_client: StubLLMClient,
) -> None:
    constraint = NoGreetingOutputConstraint(true_client, seed=0)
    assert constraint.instructions(train_or_test="test") == "挨拶抜きで本文から開始してください。"
