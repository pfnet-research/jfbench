import asyncio

from conftest import StubLLMClient

from jfbench.constraints.structure import HeadingStructureConstraint


def test_heading_structure_constraint_accepts_valid_structure(
    true_client: StubLLMClient,
) -> None:
    constraint = HeadingStructureConstraint(true_client)
    text = "Title\n\n## Section\nParagraph content."
    assert asyncio.run(constraint.evaluate(text))[0]
    assert true_client.captured_prompt is not None
    assert text in true_client.captured_prompt


def test_heading_structure_constraint_rejects_when_llm_denies(
    false_client: StubLLMClient,
) -> None:
    constraint = HeadingStructureConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("No headings here."))[0]


def test_heading_structure_constraint_accepts_prefixed_true_response(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True: headings verified"
    constraint = HeadingStructureConstraint(true_client)
    assert asyncio.run(constraint.evaluate("## Intro\nContent"))[0]


def test_heading_structure_constraint_rejects_prefixed_false_response(
    false_client: StubLLMClient,
) -> None:
    false_client.reply = "False - heading missing"
    constraint = HeadingStructureConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("Plain paragraph."))[0]
