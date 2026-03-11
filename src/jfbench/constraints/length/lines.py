import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class NewLinesLengthConstraint(ConstraintGroupMixin):
    def __init__(self, newlines: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.newlines = newlines

    def evaluate(self, value: str) -> ConstraintEvaluation:
        count = value.count("\n")
        if count != self.newlines:
            reason = f"[New Lines Length] Expected {self.newlines} newlines, found {count}."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"改行の数を{self.newlines}回にしてください。",
            f"全体で{self.newlines}個の改行を含めるようにしてください。",
            f"改行数が{self.newlines}となるよう出力を整えてください。",
            f"回答の改行回数を{self.newlines}回に固定してください。",
            f"改行が{self.newlines}個になるように行を分けてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"改行は合計で{self.newlines}回だけ入れてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"改行回数が{self.newlines}になるように調整してください。"


class BlankLinesLengthConstraint(ConstraintGroupMixin):
    def __init__(self, blank_lines: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.blank_lines = blank_lines

    def evaluate(self, value: str) -> ConstraintEvaluation:
        blank = sum(1 for line in value.splitlines() if not line.strip())
        if blank != self.blank_lines:
            reason = (
                f"[Blank Lines Length] Expected {self.blank_lines} blank lines, found {blank}."
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"空行の数を{self.blank_lines}行にしてください。",
            f"全体で{self.blank_lines}行の空行を含めるようにしてください。",
            f"空行を{self.blank_lines}行に揃えて配置してください。",
            f"回答中の空行回数を{self.blank_lines}行に固定してください。",
            f"空白行を数えて{self.blank_lines}行になるように調整してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"空行は{self.blank_lines}行だけにしてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"空行が{self.blank_lines}行になるように調整してください。"
