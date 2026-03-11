import asyncio
import logging

from _pytest.logging import LogCaptureFixture
from conftest import StubLLMClient

from jfbench.constraints.meta_output import NoSelfReferenceConstraint
from jfbench.constraints.meta_output import SelfReferenceConstraint


def test_self_reference_constraint_accepts_when_llm_confirms(true_client: StubLLMClient) -> None:
    constraint = SelfReferenceConstraint(true_client)
    assert asyncio.run(constraint.evaluate("I believe the report is complete."))[0]
    assert true_client.captured_prompt is not None
    assert "report is complete" in true_client.captured_prompt


def test_self_reference_constraint_rejects_when_llm_disagrees(
    false_client: StubLLMClient,
) -> None:
    constraint = SelfReferenceConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("The summary is concise."))[0]


def test_self_reference_constraint_logs_llm_response_on_failure(
    false_client: StubLLMClient,
    caplog: LogCaptureFixture,
) -> None:
    false_client.reply = "False - no self reference found"
    constraint = SelfReferenceConstraint(false_client)
    with caplog.at_level(logging.INFO):
        assert not asyncio.run(constraint.evaluate("Summary provided."))[0]
    assert "no self reference" in caplog.text


def test_self_reference_constraint_instructions_test_variant(
    true_client: StubLLMClient,
) -> None:
    constraint = SelfReferenceConstraint(true_client, seed=0)
    assert (
        constraint.instructions(train_or_test="test") == "自己言及の語句を一度は入れてください。"
    )


def test_no_self_reference_constraint_accepts_valid_text(true_client: StubLLMClient) -> None:
    constraint = NoSelfReferenceConstraint(true_client)
    assert asyncio.run(constraint.evaluate("The report summarizes project progress."))[0]
    assert true_client.captured_prompt is not None
    assert "project progress" in true_client.captured_prompt


def test_no_self_reference_constraint_rejects_when_llm_says_false(
    false_client: StubLLMClient,
) -> None:
    constraint = NoSelfReferenceConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("I believe the report is complete."))[0]


def test_no_self_reference_constraint_accepts_prefixed_true_response(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True: self-reference not detected"
    constraint = NoSelfReferenceConstraint(true_client)
    assert asyncio.run(constraint.evaluate("Findings focus on the data only."))[0]


def test_no_self_reference_constraint_rejects_prefixed_false_response(
    false_client: StubLLMClient,
) -> None:
    false_client.reply = "False - contains self reference"
    constraint = NoSelfReferenceConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("I think the summary is clear."))[0]


def test_no_self_reference_constraint_logs_llm_response_on_failure(
    false_client: StubLLMClient,
    caplog: LogCaptureFixture,
) -> None:
    false_client.reply = "False - self mention present"
    constraint = NoSelfReferenceConstraint(false_client)
    with caplog.at_level(logging.INFO):
        assert not asyncio.run(constraint.evaluate("I consider this task complete."))[0]
    assert "False - self mention present" in caplog.text


def test_no_self_reference_constraint_instructions_test_variant(
    true_client: StubLLMClient,
) -> None:
    constraint = NoSelfReferenceConstraint(true_client, seed=0)
    assert constraint.instructions(train_or_test="test") == "自己言及は入れずに書いてください。"
