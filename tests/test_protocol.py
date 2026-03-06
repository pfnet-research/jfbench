from typing import cast

from jfbench.protocol import Constraint
from jfbench.protocol import ConstraintEvaluation


class _DummyConstraint:
    def evaluate(self, value: str) -> ConstraintEvaluation:
        return value == "ok", None

    def instructions(self, train_or_test: str = "train") -> str:
        return "dummy instructions"

    @property
    def group(self) -> str:
        return "Test"

    def rewrite_instructions(self) -> str:
        return "rewrite dummy"

    @property
    def competitives(self) -> list[str]:
        return []

    def to_serializable_kwargs(self) -> dict[str, object]:
        return {}


def _use_constraint(constraint: Constraint) -> tuple[bool, str]:
    ok = cast("tuple[bool, str | None]", constraint.evaluate("ok"))[0]
    return bool(ok), constraint.instructions()


def test_protocol_accepts_structural_implementation() -> None:
    dummy = _DummyConstraint()
    ok, instructions = _use_constraint(dummy)
    assert ok is True
    assert instructions == "dummy instructions"
