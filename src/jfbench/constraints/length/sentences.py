import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_sentences
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class SentencesLengthConstraint(ConstraintGroupMixin):
    def __init__(
        self,
        min_sentences: int,
        max_sentences: int,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    def evaluate(self, value: str) -> ConstraintEvaluation:
        num_sentences = len(split_sentences(value))

        if num_sentences < self.min_sentences:
            reason = (
                "[Sentences Length] Number of sentences "
                f"{num_sentences} is less than minimum {self.min_sentences}"
            )
            logger.info(reason)
            return False, reason
        if num_sentences > self.max_sentences:
            reason = (
                "[Sentences Length] Number of sentences "
                f"{num_sentences} exceeds maximum {self.max_sentences}"
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"出力は{self.min_sentences}～{self.max_sentences}文に収めてください。",
            f"文数が{self.min_sentences}以上{self.max_sentences}以下になるように調整してください。",
            f"文の合計を{self.min_sentences}～{self.max_sentences}本に調整し、句点で区切られた文のみとしてください。",
            f"必ず{self.min_sentences}文以上、{self.max_sentences}文以下となるよう文を並べてください。",
            f"文章は{self.min_sentences}～{self.max_sentences}文に限定してください。",
        ]
        suffix = "句読点や改行を適切に使い、pysbdで文分割しやすい形にしてください。"
        if train_or_test == "train":
            return self._random_instruction(templates) + suffix
        if train_or_test == "test":
            return (
                f"文は{self.min_sentences}～{self.max_sentences}文にし、"
                "pysbdで区切りやすい句読点と改行に整えてください。"
            )
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        detail = "pysbdで文分割しやすいよう、句読点や改行を整えてください。"
        return (
            f"文数が{self.min_sentences}以上{self.max_sentences}以下になるようにしてください。"
            + detail
        )
