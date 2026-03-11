import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_words
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class OverlapRatioIfbenchConstraint(ConstraintGroupMixin):
    def __init__(
        self, document: str, target_ratio_percent: int, *, seed: int | None = None
    ) -> None:
        super().__init__(seed=seed)
        document = document.rstrip("\r\n")
        self.document = document
        self.target_ratio_percent = target_ratio_percent
        self._doc_trigrams = set(trigrams(document))

    def evaluate(self, value: str) -> ConstraintEvaluation:
        value_trigrams = trigrams(value)
        if not value_trigrams:
            reason = "[Overlap Ratio] No trigrams to compare."
            logger.info(reason)
            return False, reason
        overlap = sum(1 for trigram in value_trigrams if trigram in self._doc_trigrams)
        ratio = (overlap / len(value_trigrams)) * 100
        if abs(ratio - self.target_ratio_percent) > 2:
            reason = (
                f"[Overlap Ratio] Trigram overlap {ratio:.2f}% not within ±2% of "
                f"{self.target_ratio_percent}%."
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"提供された指示文との3連語重複率を{self.target_ratio_percent}%±2%に収めてください。",
            f"指示文との3-gramの重複率が{self.target_ratio_percent}%前後2%以内になるようにしてください。",
            f"入力の指示文との三連語の一致率を{self.target_ratio_percent}%±2%に維持してください。",
            f"与えられた指示文との三連語ベースの重複率を{self.target_ratio_percent}%前後2%の範囲にしてください。",
            f"指示文との3語連続の重複率を{self.target_ratio_percent}%±2%に調整してください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"三連語の重複率が{self.target_ratio_percent}%±2%内に収まるように調整してください。"


def trigrams(text: str) -> list[str]:
    words = split_words(text)
    return [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
