import asyncio

from conftest import StubLLMClient

from jfbench.constraints.structure import SectionStructureConstraint


def test_section_structure_constraint_accepts_valid_structure(
    true_client: StubLLMClient,
) -> None:
    constraint = SectionStructureConstraint(true_client)
    text = "Introduction\n...\nBody\n...\nConclusion\n..."
    assert asyncio.run(constraint.evaluate(text))[0]
    assert true_client.captured_prompt is not None
    assert "Introduction" in true_client.captured_prompt


def test_section_structure_constraint_rejects_when_llm_denies(
    false_client: StubLLMClient,
) -> None:
    constraint = SectionStructureConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("Random text without structure."))[0]


def test_section_structure_constraint_accepts_prefixed_true_response(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True: sections aligned"
    constraint = SectionStructureConstraint(true_client)
    assert asyncio.run(constraint.evaluate("Introduction\nBody\nConclusion"))[0]


def test_section_structure_constraint_rejects_prefixed_false_response(
    false_client: StubLLMClient,
) -> None:
    false_client.reply = "False - lacks sections"
    constraint = SectionStructureConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("Only one block."))[0]
