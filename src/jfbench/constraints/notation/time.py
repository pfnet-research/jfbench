import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class TimeNotationConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        if not re.search(r"\b([01]\d|2[0-3]):[0-5]\d(:[0-5]\d)?\b", value):
            reason = "[Time Notation] Time must use 24h format HH:MM or HH:MM:SS."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力には24時間表記のHH:MMまたはHH:MM:SS形式の時刻を少なくとも1つ含めてください。",
            "24時間制で、HH:MMかHH:MM:SSの時刻を必ず1件以上入れてください。",
            "時間表記をHH:MMまたはHH:MM:SS（24時間制）にした時刻を少なくとも一つ示してください。",
            "24hフォーマットのHH:MM/HH:MM:SSで書かれた時刻を1件以上含めてください。",
            "時刻は24時間のHH:MM（必要ならHH:MM:SS）形式で、最低1つは出力してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "24時間表記のHH:MMかHH:MM:SSを入れてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "出力を修正し、24時間表記のHH:MMまたはHH:MM:SSの時刻を少なくとも1つ含めてください。"
