from typing import cast
from typing import TYPE_CHECKING

from jfbench.constraints.content.excluded import AbstractExcludedContentConstraint
from jfbench.constraints.content.excluded import KeywordExcludedContentConstraint


if TYPE_CHECKING:
    from jfbench.llm import LLMClient


def test_keyword_excluded_instructions_test_differs_from_rewrite() -> None:
    keywords = ["禁止", "回避"]
    constraint = KeywordExcludedContentConstraint(keywords)
    instruction = constraint.instructions("test")
    assert instruction != constraint.rewrite_instructions()
    for keyword in keywords:
        assert keyword in instruction


def test_abstract_excluded_instructions_test_differs_from_rewrite() -> None:
    content = "非公開事項"
    client = cast("LLMClient", object())
    constraint = AbstractExcludedContentConstraint(
        client=client,
        document="doc",
        content=content,
    )
    instruction = constraint.instructions("test")
    assert instruction != constraint.rewrite_instructions()
    assert content in instruction
