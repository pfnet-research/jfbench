import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class CharactersLengthConstraint(ConstraintGroupMixin):
    def __init__(
        self,
        min_length: int,
        max_length: int,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.min_length = min_length
        self.max_length = max_length

    def evaluate(self, value: str) -> ConstraintEvaluation:
        length = len(value)
        if length < self.min_length:
            reason = f"[Characters Length] Length {length} is less than minimum {self.min_length}"
            logger.info(reason)
            return False, reason
        if length > self.max_length:
            reason = f"[Characters Length] Length {length} exceeds maximum {self.max_length}"
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"出力は{self.min_length}文字以上{self.max_length}文字以下に収めてください。",
            f"文字数を{self.min_length}～{self.max_length}の範囲に制限して回答してください。",
            f"全体の長さが{self.min_length}文字未満にも{self.max_length}文字超にもならないよう調整してください。",
            f"内容を調整し、{self.min_length}から{self.max_length}文字の間で完結させてください。",
            f"回答文字数は最低{self.min_length}、最大{self.max_length}までにしてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"文字数は{self.min_length}字以上{self.max_length}字以下にしてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"文字数が{self.min_length}以上{self.max_length}以下に収まるように内容を調整してください。"
