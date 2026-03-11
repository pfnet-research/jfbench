import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class SectionsLengthConstraint(ConstraintGroupMixin):
    def __init__(
        self,
        min_sections: int,
        max_sections: int,
        sections_delimiter: str = "\n\n",
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.min_sections = min_sections
        self.max_sections = max_sections
        if not sections_delimiter:
            raise ValueError("sections_delimiter must be a non-empty string")
        self.sections_delimiter = sections_delimiter

    def evaluate(self, value: str) -> ConstraintEvaluation:
        sections = [s for s in value.strip().split(self.sections_delimiter) if s.strip()]
        num_sections = len(sections)
        if num_sections < self.min_sections:
            reason = (
                "[Sections Length] Number of sections "
                f"{num_sections} is less than minimum {self.min_sections}"
            )
            logger.info(reason)
            return False, reason
        if num_sections > self.max_sections:
            reason = (
                "[Sections Length] Number of sections "
                f"{num_sections} exceeds maximum {self.max_sections}"
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            (
                f"出力は{self.min_sections}～{self.max_sections}個のセクションで構成し、"
                f"区切りには「{self.sections_delimiter!r}」を使ってください。"
            ),
            (
                f"セクション数を{self.min_sections}以上{self.max_sections}以下に制御し、"
                f"各セクションを{self.sections_delimiter!r}で区切ってください。"
            ),
            (
                f"{self.sections_delimiter!r}を境界として最低{self.min_sections}、最大{self.max_sections}の"
                "セクションに分割した出力にしてください。"
            ),
            (
                f"回答は{self.min_sections}～{self.max_sections}セクションでまとめ、"
                f"セクション間は必ず{self.sections_delimiter!r}を挟んでください。"
            ),
            (
                f"セクションを{self.min_sections}から{self.max_sections}の範囲に収め、"
                f"区切り記号として{self.sections_delimiter!r}のみを使用してください。"
            ),
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return (
                f"セクションは{self.min_sections}～{self.max_sections}個にして、"
                f"区切りは{self.sections_delimiter!r}で統一してください。"
            )
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            f"セクション区切り（{self.sections_delimiter!r}）を調整し、セクション数が"
            f"{self.min_sections}以上{self.max_sections}以下になるように書き直してください。"
        )
