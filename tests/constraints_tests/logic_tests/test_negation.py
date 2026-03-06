import asyncio
import random

from jfbench.constraints.logic import DoubleNegationLogicConstraint
from jfbench.constraints.logic import NegationLogicConstraint


class DummyConstraint:
    def __init__(self, result: bool, *, instructions_text: str = "") -> None:
        self._result = result
        self._instructions_text = instructions_text or "dummy instruction"
        self.last_value: str | None = None

    def evaluate(self, value: str) -> tuple[bool, str | None]:
        self.last_value = value
        return self._result, None

    def instructions(self, train_or_test: str = "train") -> str:
        return self._instructions_text

    @property
    def group(self) -> str:
        return "Test"

    def rewrite_instructions(self) -> str:
        return f"Do not satisfy: {self._instructions_text}"

    @property
    def competitives(self) -> list[str]:
        return []

    def to_serializable_kwargs(self) -> dict[str, object]:
        return {}


def test_negation_constraint_inverts_positive_constraint() -> None:
    positive = DummyConstraint(result=True, instructions_text="Provide a number")
    constraint = NegationLogicConstraint(positive, seed=0)

    passed, reason = asyncio.run(constraint.evaluate("42"))

    assert not passed
    assert reason is not None
    assert "[Negation Logic]" in reason
    assert positive.last_value == "42"


def test_negation_constraint_passes_when_positive_fails() -> None:
    positive = DummyConstraint(result=False, instructions_text="Provide a number")
    constraint = NegationLogicConstraint(positive, seed=1)

    passed, reason = asyncio.run(constraint.evaluate("hello"))

    assert passed
    assert reason is None
    assert positive.last_value == "hello"


def test_negation_constraint_instructions_wrap_positive() -> None:
    positive = DummyConstraint(result=False, instructions_text="Provide a number")
    seed = 2
    constraint = NegationLogicConstraint(positive, seed=seed)
    templates = [
        f"次の制約をわざと満たさない内容にしてください: {positive.instructions()}",
        f"以下の条件が成立しないように回答してください: {positive.instructions()}",
        f"与えられた制約を破る形になるように出力を作成してください: {positive.instructions()}",
        f"指定の制約をあえて失敗させる回答にしてください: {positive.instructions()}",
        f"次の制約が成立しない出力を選んでください: {positive.instructions()}",
    ]

    expected_instruction = random.Random(seed).choice(templates)

    assert constraint.instructions() == expected_instruction


def test_negation_constraint_instructions_test_variant() -> None:
    positive = DummyConstraint(result=False, instructions_text="Provide a number")
    constraint = NegationLogicConstraint(positive, seed=0)

    assert (
        constraint.instructions(train_or_test="test")
        == "次の制約は満たさないでください: Provide a number"
    )


def test_double_negation_constraint_passes_when_positive_succeeds() -> None:
    positive = DummyConstraint(result=True, instructions_text="Stay compliant")
    constraint = DoubleNegationLogicConstraint(positive, seed=3)

    passed, reason = asyncio.run(constraint.evaluate("ok"))

    assert passed
    assert reason is None
    assert positive.last_value == "ok"


def test_double_negation_constraint_fails_when_positive_fails() -> None:
    positive = DummyConstraint(result=False, instructions_text="Stay compliant")
    constraint = DoubleNegationLogicConstraint(positive, seed=4)

    passed, reason = asyncio.run(constraint.evaluate("nope"))

    assert not passed
    assert reason is not None
    assert "[Double Negation Logic]" in reason
    assert positive.last_value == "nope"


def test_double_negation_constraint_instructions_use_templates() -> None:
    positive = DummyConstraint(result=True, instructions_text="Keep it safe")
    seed = 5
    constraint = DoubleNegationLogicConstraint(positive, seed=seed)
    templates = [
        f"次の制約が成立しない状態を避け、そのまま満たしてください: {positive.instructions()}",
        f"以下の条件を否定せず、素直に満たす回答にしてください: {positive.instructions()}",
        f"与えられた制約を崩さずに満たす内容で回答してください: {positive.instructions()}",
        f"指定の制約をそのまま成立させるように書いてください: {positive.instructions()}",
        f"次の制約を無効化せず、その条件に従ってください: {positive.instructions()}",
    ]

    expected_instruction = random.Random(seed).choice(templates)

    assert constraint.instructions() == expected_instruction


def test_double_negation_constraint_instructions_test_variant() -> None:
    positive = DummyConstraint(result=True, instructions_text="Keep it safe")
    constraint = DoubleNegationLogicConstraint(positive, seed=0)

    assert (
        constraint.instructions(train_or_test="test")
        == "次の制約が成立しないことはないようにしてください: Keep it safe"
    )
