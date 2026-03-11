import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class TitlecaseNotationConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        if not re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", value):
            reason = "[Titlecase Notation] No Title Case phrase detected."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "少なくとも一つのTitle Caseのフレーズを含めてください。",
            "単語ごとに頭文字が大文字のTitle Case表現を入れてください。",
            "Title Caseのフレーズを文章中に1つ以上入れてください。",
            "頭文字が大文字になるTitle Caseの語句を含めてください。",
            "Title Caseの単語列を最低1箇所に挿入してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "Title Caseの語句を含めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "Title Caseのフレーズを追加してください。"
