from typing import cast
from typing import TYPE_CHECKING

from jfbench.constraints.content.reason import NoReasonContentConstraint
from jfbench.constraints.content.reason import ReasonContentConstraint


if TYPE_CHECKING:
    from jfbench.llm import LLMClient


def test_no_reason_instructions_test_differs_from_rewrite() -> None:
    client = cast("LLMClient", object())
    constraint = NoReasonContentConstraint(client=client)
    assert constraint.instructions("test") != constraint.rewrite_instructions()


def test_reason_instructions_test_differs_from_rewrite() -> None:
    client = cast("LLMClient", object())
    constraint = ReasonContentConstraint(client=client, document="doc")
    assert constraint.instructions("test") != constraint.rewrite_instructions()
