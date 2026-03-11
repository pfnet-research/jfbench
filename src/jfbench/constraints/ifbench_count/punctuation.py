import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class PunctuationCountIfbenchConstraint(ConstraintGroupMixin):
    def __init__(self, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._required_chars: tuple[str, ...] = ("。", "、", "；", "：", "？", "！", "？！")

    def evaluate(self, value: str) -> ConstraintEvaluation:
        missing = [char for char in self._required_chars if char not in value]
        if missing:
            reason = (
                f"[Punctuation Count] Missing required punctuation marks: {', '.join(missing)}."
            )
            logger.info(reason)
            return False, reason
        if "（" not in value or "）" not in value:
            reason = "[Punctuation Count] Parentheses are required at least once."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "回答には「。」「、」「；」「：」「？」「！」をそれぞれ1回以上使い、全角の括弧（　）とインターロバン（？！）も一度入れてください。",
            "全角の句読点（。 、 ； ： ？ ！）を網羅し、?! ではなく？！の形と全角括弧も必ず含めてください。",
            "句読点の種類を一通り登場させ、特に；と：、全角の？！、そして全角の括弧（　）を漏らさないでください。",
            "標準的な全角句読点を抜かさず使い、インターロバンとして？！と全角括弧も一度入れてください。",
            "「。」「、」などの全角句読点を最低1回ずつ使い、？！と全角の括弧も併せて含めてください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "指定された全角句読点（特に；、：、？！）と全角の括弧をすべて含むように追記してください。"
