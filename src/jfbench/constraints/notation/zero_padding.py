import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class ZeroPaddingNotationConstraint(ConstraintGroupMixin):
    def __init__(self, width: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.width = width

    def evaluate(self, value: str) -> ConstraintEvaluation:
        if self.width <= 0:
            raise ValueError("width must be positive")

        pattern = rf"\b0\d{{{self.width - 1}}}\b"
        if not re.search(pattern, value):
            reason = (
                "[Zero Padding Notation] No zero-padded number "
                f"with exactly {self.width} digits found."
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"出力には少なくとも1つ、桁数{self.width}を満たすように左側をゼロで埋めた数値を含めてください。",
            f"{self.width}桁になるよう先頭にゼロを付けた数値を必ず1件以上入れてください。",
            f"ゼロで左側を埋めて合計{self.width}桁にした数字を少なくとも一つ示してください。",
            f"数値をゼロパディングし、{self.width}桁表記で示した例を最低1つは出力してください。",
            f"{self.width}桁に揃うまで先頭に0を補った値を必ず1件以上含めてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"{self.width}桁のゼロ埋め数値を含めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"出力を修正し、少なくとも1つの数値を{self.width}桁になるようゼロ埋めしてください。"
