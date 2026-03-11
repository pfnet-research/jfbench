from itertools import pairwise
import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_sentences
from jfbench.constraints._utils import split_words
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class IncrementSentenceIfbenchConstraint(ConstraintGroupMixin):
    def __init__(self, increment: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.increment = increment

    def evaluate(self, value: str) -> ConstraintEvaluation:
        sentences = split_sentences(value)
        if len(sentences) < 2:
            reason = "[Increment Sentence] At least two sentences are required."
            logger.info(reason)
            return False, reason
        word_counts = [len(split_words(sentence)) for sentence in sentences]
        for previous, current in pairwise(word_counts):
            if current - previous != self.increment:
                reason = (
                    "[Increment Sentence] Consecutive sentences must increase by "
                    f"{self.increment} words exactly."
                )
                logger.info(reason)
                return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"各文は前の文より正確に{self.increment}語多くしてください。",
            f"連続する文の語数差を常に{self.increment}語にしてください。",
            f"文ごとに{self.increment}語ずつ語数が増えるようにしてください。",
            f"先行文より{self.increment}語多い文章を続けてください。",
            f"連続する各文の語数を{self.increment}語ずつ増やしてください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"各文が前文より{self.increment}語多くなるように語数を調整してください。"
