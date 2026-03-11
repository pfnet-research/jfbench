import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class SnakecaseNotationConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        if not re.search(r"\b[a-z]+(?:_[a-z]+)+\b", value):
            reason = "[Snakecase Notation] No snake_case phrase detected."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "snake_caseのフレーズを少なくとも1つ入れてください。",
            "単語をアンダースコアで繋いだsnake_case表現を含めてください。",
            "snake_case形式の語を文章に混ぜてください。",
            "スネークケース（snake_case、例: foo_bar）の単語列を1つ以上入れてください。",
            "アンダースコア区切りのsnake_case語を必ず含めてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "snake_caseの語を含めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "snake_caseの語を文章に挿入してください。"
