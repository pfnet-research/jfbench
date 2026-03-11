import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class ParagraphsLengthConstraint(ConstraintGroupMixin):
    def __init__(
        self,
        min_paragraphs: int,
        max_paragraphs: int,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.min_paragraphs = min_paragraphs
        self.max_paragraphs = max_paragraphs

    def evaluate(self, value: str) -> ConstraintEvaluation:
        paragraphs = [p for p in value.split("\n\n") if p.strip()]
        num_paragraphs = len(paragraphs)

        if num_paragraphs < self.min_paragraphs:
            reason = (
                "[Paragraphs Length] Number of paragraphs "
                f"{num_paragraphs} is less than minimum {self.min_paragraphs}"
            )
            logger.info(reason)
            return False, reason
        if num_paragraphs > self.max_paragraphs:
            reason = (
                "[Paragraphs Length] Number of paragraphs "
                f"{num_paragraphs} exceeds maximum {self.max_paragraphs}"
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            (
                f"出力は{self.min_paragraphs}～{self.max_paragraphs}段落で構成し、"
                "段落は連続した空行（\\n\\n）で区切ってください。"
            ),
            (
                f"段落数が最低{self.min_paragraphs}、最大{self.max_paragraphs}になるようにし、"
                "段落境界は2連続の改行で示してください。"
            ),
            (
                f"\\n\\nごとに区切られた段落を{self.min_paragraphs}～{self.max_paragraphs}個になるように調整してください。"
            ),
            (
                f"回答全体を{self.min_paragraphs}以上{self.max_paragraphs}以下の段落に分け、"
                "段落間は空行2つで分離してください。"
            ),
            (
                f"最終的な文章は{self.min_paragraphs}から{self.max_paragraphs}の段落数に収め、"
                "段落の境界には必ず\\n\\nを入れてください。"
            ),
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return (
                f"段落数は{self.min_paragraphs}～{self.max_paragraphs}にし、"
                "区切りは\\n\\nでお願いします。"
            )
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"段落数が{self.min_paragraphs}以上{self.max_paragraphs}以下になるように改行区切りを調整してください。"
