import logging

from janome.tokenizer import Tokenizer

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class WordsLengthConstraint(ConstraintGroupMixin):
    def __init__(self, minimum: int, maximum: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.minimum = minimum
        self.maximum = maximum
        self._tokenizer = Tokenizer()

    def evaluate(self, value: str) -> ConstraintEvaluation:
        tokens = list(self._tokenizer.tokenize(value))
        count = len(tokens)
        if count < self.minimum:
            reason = f"[Word Length] Too few words ({count}); minimum is {self.minimum}."
            logger.info(reason)
            return False, reason
        if count > self.maximum:
            reason = f"[Word Length] Too many words ({count}); maximum is {self.maximum}."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"語数を{self.minimum}〜{self.maximum}語の範囲に収めてください。",
            f"回答の語数は{self.minimum}語以上{self.maximum}語以下にしてください。",
            f"{self.minimum}〜{self.maximum}語の間の長さに調整してください。",
            f"文章全体を{self.minimum}語から{self.maximum}語の語数でまとめてください。",
            f"{self.minimum}語未満や{self.maximum}語超過にならないようにしてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"語数は{self.minimum}語以上{self.maximum}語以下に揃えてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"語数が{self.minimum}〜{self.maximum}語に収まるようにしてください。"
