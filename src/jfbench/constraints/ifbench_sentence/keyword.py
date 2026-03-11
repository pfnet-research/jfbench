import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_sentences
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class KeywordSentenceIfbenchConstraint(ConstraintGroupMixin):
    def __init__(self, sentence_index: int, keyword: str, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.sentence_index = sentence_index
        self.keyword = keyword

    def evaluate(self, value: str) -> ConstraintEvaluation:
        sentences = split_sentences(value)
        if len(sentences) < self.sentence_index:
            reason = "[Keyword Sentence] Not enough sentences to place the keyword."
            logger.info(reason)
            return False, reason
        target = sentences[self.sentence_index - 1]
        if self.keyword.lower() not in target.lower():
            reason = (
                f"[Keyword Sentence] Keyword {self.keyword!r} not found in sentence "
                f"{self.sentence_index}."
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"{self.sentence_index}文目にキーワード「{self.keyword}」を簡潔に含めてください。",
            f"{self.keyword}を{self.sentence_index}番目の文の中で触れてください。",
            f"{self.sentence_index}番目の文にキーワード{self.keyword}を入れてください。",
            f"{self.sentence_index}文目で{self.keyword}という語を登場させてください。",
            f"{self.keyword}を{self.sentence_index}文目に短く挿入してください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"{self.sentence_index}文目に{self.keyword}が含まれるように追記してください。"
