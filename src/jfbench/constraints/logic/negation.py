import inspect
import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import Constraint
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class NegationLogicConstraint(ConstraintGroupMixin):
    def __init__(self, positive_constraint: Constraint, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.positive_constraint = positive_constraint

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        result = self.positive_constraint.evaluate(value)
        if inspect.isawaitable(result):
            result = await result
        passed, reason = result if isinstance(result, tuple) else (bool(result), None)
        if passed:
            message = "[Negation Logic] Output satisfies the positive constraint and should instead fail it."
            logger.info(message)
            return False, message
        return True, reason

    def instructions(self, train_or_test: str = "train") -> str:
        positive_instructions = self.positive_constraint.instructions(train_or_test=train_or_test)
        templates = [
            f"次の制約をわざと満たさない内容にしてください: {positive_instructions}",
            f"以下の条件が成立しないように回答してください: {positive_instructions}",
            f"与えられた制約を破る形になるように出力を作成してください: {positive_instructions}",
            f"指定の制約をあえて失敗させる回答にしてください: {positive_instructions}",
            f"次の制約が成立しない出力を選んでください: {positive_instructions}",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"次の制約は満たさないでください: {positive_instructions}"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            f"指定された制約をあえて破る形になるように出力を調整してください: "
            f"{self.positive_constraint.instructions()}"
        )


class DoubleNegationLogicConstraint(ConstraintGroupMixin):
    def __init__(self, positive_constraint: Constraint, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.positive_constraint = positive_constraint

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        result = self.positive_constraint.evaluate(value)
        if inspect.isawaitable(result):
            result = await result
        passed, reason = result if isinstance(result, tuple) else (bool(result), None)
        if not passed:
            message = "[Double Negation Logic] Output should satisfy the underlying constraint."
            logger.info(message)
            return False, message
        return True, reason

    def instructions(self, train_or_test: str = "train") -> str:
        positive_instructions = self.positive_constraint.instructions(train_or_test=train_or_test)
        templates = [
            f"次の制約が成立しない状態を避け、そのまま満たしてください: {positive_instructions}",
            f"以下の条件を否定せず、素直に満たす回答にしてください: {positive_instructions}",
            f"与えられた制約を崩さずに満たす内容で回答してください: {positive_instructions}",
            f"指定の制約をそのまま成立させるように書いてください: {positive_instructions}",
            f"次の制約を無効化せず、その条件に従ってください: {positive_instructions}",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"次の制約が成立しないことはないようにしてください: {positive_instructions}"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return self.positive_constraint.rewrite_instructions()
