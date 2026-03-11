import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class DateNotationConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        if not re.search(r"\b\d{4}年\d{2}月\d{2}日\b", value):
            reason = "[Date Notation] Date format YYYY年MM月DD日 is missing."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力には少なくとも1つYYYY年MM月DD日で表記した日付を含めてください。",
            "少なくとも1件、YYYY年MM月DD日（西暦4桁-月2桁-日2桁）形式の日付を書いてください。",
            "回答に日付を1つ以上入れ、その表現をYYYY年MM月DD日に統一してください。",
            "YYYY年MM月DD日フォーマットの日付を少なくとも一つ出力に加えてください。",
            "4桁-2桁-2桁（YYYY年MM月DD日）の日付を最低1件は盛り込んでください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "YYYY年MM月DD日の日付を含めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "出力にYYYY年MM月DD日形式の日付を少なくとも一つ追加してください。"
