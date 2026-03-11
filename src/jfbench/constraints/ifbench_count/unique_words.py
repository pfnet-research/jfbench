import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_words
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class UniqueWordCountIfbenchConstraint(ConstraintGroupMixin):
    def __init__(self, minimum_unique: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.minimum_unique = minimum_unique

    def evaluate(self, value: str) -> ConstraintEvaluation:
        unique_words = set(split_words(value))
        if len(unique_words) < self.minimum_unique:
            reason = (
                "[Unique Word Count] Not enough unique words. "
                f"Expected at least {self.minimum_unique}, found {len(unique_words)}."
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"回答には少なくとも{self.minimum_unique}種類の固有の単語を使用してください。",
            f"単語の重複を避け、{self.minimum_unique}語以上の異なる語彙で構成してください。",
            f"{self.minimum_unique}種類以上のユニークな単語を盛り込んでください。",
            f"同じ単語を繰り返し過ぎないようにし、{self.minimum_unique}語以上の語彙を用いてください。",
            f"ユニークな語彙数が{self.minimum_unique}以上になるよう工夫してください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"使われている単語の種類が{self.minimum_unique}以上になるように書き換えてください。"
