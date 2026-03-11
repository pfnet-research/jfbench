import asyncio
import logging

from _pytest.logging import LogCaptureFixture
from conftest import StubLLMClient

from jfbench.constraints.meta_output import NoSelfAttestationConstraint
from jfbench.constraints.meta_output import SelfAttestationConstraint


def test_no_self_attestation_constraint_accepts_when_llm_confirms(
    true_client: StubLLMClient,
) -> None:
    constraint = NoSelfAttestationConstraint(true_client)
    assert asyncio.run(constraint.evaluate("Here is the result you asked for."))[0]
    assert true_client.captured_prompt is not None
    assert "result you asked for" in true_client.captured_prompt


def test_no_self_attestation_constraint_rejects_when_llm_disagrees(
    false_client: StubLLMClient,
) -> None:
    constraint = NoSelfAttestationConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("All constraints are satisfied."))[0]


def test_no_self_attestation_constraint_logs_llm_response_on_failure(
    false_client: StubLLMClient,
    caplog: LogCaptureFixture,
) -> None:
    false_client.reply = "False - self attestation present"
    constraint = NoSelfAttestationConstraint(false_client)
    with caplog.at_level(logging.INFO):
        assert not asyncio.run(constraint.evaluate("I complied with every rule."))[0]
    assert "self attestation present" in caplog.text


def test_no_self_attestation_constraint_instructions_test_variant(
    true_client: StubLLMClient,
) -> None:
    constraint = NoSelfAttestationConstraint(true_client, seed=0)
    assert (
        constraint.instructions(train_or_test="test")
        == "制約を満たしたと明言する文は入れないでください。"
    )


def test_self_attestation_constraint_accepts_statement(true_client: StubLLMClient) -> None:
    constraint = SelfAttestationConstraint(true_client)
    assert asyncio.run(constraint.evaluate("All constraints have been satisfied as requested."))[0]
    assert true_client.captured_prompt is not None
    assert "制約" in true_client.captured_prompt


def test_self_attestation_constraint_rejects_when_llm_disagrees(
    false_client: StubLLMClient,
) -> None:
    constraint = SelfAttestationConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("Here is the answer without confirmation."))[0]


def test_self_attestation_constraint_accepts_prefixed_true_response(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True - attestation present"
    constraint = SelfAttestationConstraint(true_client)
    assert asyncio.run(constraint.evaluate("All checks complete as confirmed."))[0]


def test_self_attestation_constraint_rejects_prefixed_false_response(
    false_client: StubLLMClient,
) -> None:
    false_client.reply = "False: confirmation missing"
    constraint = SelfAttestationConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("Deliverable is attached."))[0]


def test_self_attestation_constraint_logs_llm_response_on_failure(
    false_client: StubLLMClient,
    caplog: LogCaptureFixture,
) -> None:
    false_client.reply = "False - attestation absent"
    constraint = SelfAttestationConstraint(false_client)
    with caplog.at_level(logging.INFO):
        assert not asyncio.run(constraint.evaluate("Here is the solution."))[0]
    assert "False - attestation absent" in caplog.text


def test_self_attestation_constraint_instructions_test_variant(
    true_client: StubLLMClient,
) -> None:
    constraint = SelfAttestationConstraint(true_client, seed=0)
    assert (
        constraint.instructions(train_or_test="test")
        == "制約を満たしたと明言する文を含めてください。"
    )
