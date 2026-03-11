from typing import cast
from typing import TYPE_CHECKING

from jfbench.constraints.content.irrevant import IrrevantContentConstraint


if TYPE_CHECKING:
    from jfbench.llm import LLMClient


def test_irrevant_instructions_test_differs_from_rewrite() -> None:
    client = cast("LLMClient", object())
    constraint = IrrevantContentConstraint(client=client, document="doc")
    assert constraint.instructions("test") != constraint.rewrite_instructions()
