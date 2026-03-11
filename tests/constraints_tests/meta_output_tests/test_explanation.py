import asyncio
import logging

from _pytest.logging import LogCaptureFixture
from conftest import StubLLMClient

from jfbench.constraints.meta_output import ExplanationConstraint
from jfbench.constraints.meta_output import NoExplanationConstraint


def test_explanation_constraint_accepts_when_llm_confirms(true_client: StubLLMClient) -> None:
    constraint = ExplanationConstraint(true_client)
    assert asyncio.run(constraint.evaluate("Answer: 42 because it fits the riddle."))[0]
    assert true_client.captured_prompt is not None
    assert "Answer" in true_client.captured_prompt


def test_explanation_constraint_rejects_when_llm_disagrees(
    false_client: StubLLMClient,
) -> None:
    constraint = ExplanationConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("Answer: 42"))[0]


def test_explanation_constraint_logs_llm_response_on_failure(
    false_client: StubLLMClient, caplog: LogCaptureFixture
) -> None:
    false_client.reply = "False - explanation missing"
    constraint = ExplanationConstraint(false_client)
    with caplog.at_level(logging.INFO):
        assert not asyncio.run(constraint.evaluate("Result: 10"))[0]
    assert "explanation missing" in caplog.text


def test_explanation_constraint_instructions_test_variant(
    true_client: StubLLMClient,
) -> None:
    constraint = ExplanationConstraint(true_client, seed=0)
    assert constraint.instructions(train_or_test="test") == "末尾に短い説明文を付けてください。"


def test_no_explanation_constraint_accepts_clean_output(true_client: StubLLMClient) -> None:
    constraint = NoExplanationConstraint(true_client)
    assert asyncio.run(constraint.evaluate("Final answer: 42"))[0]
    assert true_client.captured_prompt is not None
    assert "Final answer" in true_client.captured_prompt


def test_no_explanation_constraint_rejects_when_llm_detects_issue(
    false_client: StubLLMClient,
) -> None:
    constraint = NoExplanationConstraint(false_client)
    assert not asyncio.run(
        constraint.evaluate("Final answer: 42. Because I computed it earlier.")
    )[0]


def test_no_explanation_constraint_accepts_prefixed_true_response(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True: no explanation"
    constraint = NoExplanationConstraint(true_client)
    assert asyncio.run(constraint.evaluate("Output: value=7"))[0]


def test_no_explanation_constraint_rejects_prefixed_false_response(
    false_client: StubLLMClient,
) -> None:
    false_client.reply = "False - extra reasoning detected"
    constraint = NoExplanationConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("Answer is 5 because reasons."))[0]


def test_no_explanation_constraint_logs_llm_response_on_failure(
    false_client: StubLLMClient,
    caplog: LogCaptureFixture,
) -> None:
    false_client.reply = "False - lingering explanation"
    constraint = NoExplanationConstraint(false_client)
    with caplog.at_level(logging.INFO):
        assert not asyncio.run(constraint.evaluate("I must explain this result."))[0]
    assert "False - lingering explanation" in caplog.text


def test_no_explanation_constraint_instructions_test_variant(
    true_client: StubLLMClient,
) -> None:
    constraint = NoExplanationConstraint(true_client, seed=0)
    assert (
        constraint.instructions(train_or_test="test") == "最後に説明文を付けずに終えてください。"
    )
