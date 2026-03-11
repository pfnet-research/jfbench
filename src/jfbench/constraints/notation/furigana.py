import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class FuriganaNotationConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        if not re.search(r"<ruby>[^<]+<rt>[^<]+</rt></ruby>", value, re.DOTALL):
            reason = "[Furigana Notation] Missing <ruby> tag for furigana."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "回答には少なくとも1箇所、漢字に対するふりがなをHTMLの<ruby>タグで付けてください。",
            "少なくとも一つの漢字に<ruby>タグを使ったふりがなを振ってください。",
            "ルビ用の<ruby>...</ruby>タグで読みを示した箇所を1つ以上含めてください。",
            "漢字表記に<ruby>タグでふりがなを添えた例を最低1箇所は入れてください。",
            "回答内の漢字に対し<ruby>タグを用いたふりがなを少なくとも一度付けてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "回答内に<ruby>タグのふりがなを入れてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "出力を修正し、<ruby>タグを用いたふりがなを少なくとも1箇所に付記してください。"
