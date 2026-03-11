import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class NumbersCountIfbenchConstraint(ConstraintGroupMixin):
    def __init__(self, expected_count: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.expected_count = expected_count

    def evaluate(self, value: str) -> ConstraintEvaluation:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
        if len(numbers) != self.expected_count:
            reason = (
                "[Numbers Count] Number of numeric values does not match. "
                f"Expected {self.expected_count}, found {len(numbers)}."
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"回答には正確に{self.expected_count}個の数値だけを含めてください。",
            f"数値の出現回数を{self.expected_count}個ちょうどにしてください。",
            f"{self.expected_count}個の数値を含め、それ以上も以下も使わないでください。",
            f"文中の数値は合計{self.expected_count}個にそろえてください。",
            f"数値の数が{self.expected_count}個になるように調整してください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"数値の出現回数を{self.expected_count}個に合わせてください。"
