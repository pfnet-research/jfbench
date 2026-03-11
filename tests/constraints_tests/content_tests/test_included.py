from typing import cast
from typing import TYPE_CHECKING

from jfbench.constraints.content.included import AbstractIncludedContentConstraint
from jfbench.constraints.content.included import KeywordIncludedContentConstraint


if TYPE_CHECKING:
    from jfbench.llm import LLMClient


def test_keyword_included_instructions_test_differs_from_rewrite() -> None:
    keywords = {"猫": 2, "犬": 1}
    constraint = KeywordIncludedContentConstraint(keywords)
    instruction = constraint.instructions("test")
    assert instruction != constraint.rewrite_instructions()
    for keyword in keywords:
        assert keyword in instruction


def test_abstract_included_instructions_test_differs_from_rewrite() -> None:
    content = "重要な点"
    client = cast("LLMClient", object())
    constraint = AbstractIncludedContentConstraint(
        client=client,
        document="doc",
        content=content,
    )
    instruction = constraint.instructions("test")
    assert instruction != constraint.rewrite_instructions()
    assert content in instruction
