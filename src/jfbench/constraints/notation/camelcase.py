import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class CamelcaseNotationConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        if not re.search(r"\b(?:[A-Z][a-z0-9]+){2,}\b", value):
            reason = "[Camecase Notation] No CamelCase phrase detected."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "少なくとも一つのCamelCaseフレーズを含めてください。",
            "CamelCase形式の語を文章に1つ以上入れてください。",
            "大文字で単語を区切ったCamelCaseのフレーズを含めてください。",
            "CamelCase（例: MyExampleWord）の語を挿入してください。",
            "CamelCase（キャメルケース）表現を少なくとも一箇所に入れてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "CamelCaseの語を入れてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "CamelCaseの語を追加してください。"
