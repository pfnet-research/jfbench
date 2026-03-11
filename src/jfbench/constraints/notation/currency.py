import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class CurrencyNotationConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        if not re.search(r"(?<!\d)¥(?:0|[1-9]\d{0,2}(?:,\d{3})*)(?![\d.])", value):
            reason = "[Currency Notation] Currency formatting with leading ¥ and comma separators is missing."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力には少なくとも1つ、先頭に¥を付けた3桁区切りの整数金額を含めてください。",
            "通貨表記は¥を前置し、小数を使わず3桁ごとにカンマを入れた金額を1件以上示してください。",
            "¥マーク付きでカンマ区切りの整数金額（例: ¥1,234）を少なくとも一つ記載してください。",
            "金額は¥1,234のように先頭¥と3桁ごとのカンマで示し、必ず1件は出力に含めてください。",
            "通貨は円表示とし、¥付きの3桁カンマ区切りの整数金額を最低1つ盛り込んでください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "¥から始まり3桁ごとにカンマで区切った金額(e.g. ¥1,234)を入れてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "出力に¥1,234のようなカンマ区切りの金額を少なくとも1件追加してください。"
