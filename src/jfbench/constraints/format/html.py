import logging
import re

import html5lib

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class HtmlFormatConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        try:
            unwrapped = _unwrap_markdown_code_fence(value)
            html5lib.HTMLParser(strict=True).parse(unwrapped)
            return True, None
        except Exception as e:
            reason = f"[HTML] Validation failed: {e}"
            logger.info(reason)
            return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力全体を有効なHTML5文書として記述してください。",
            "HTMLタグで構成された完全なHTML5文書を生成し、不正な構造を避けてください。",
            "正しいHTML5構文のみを用い、単体でパースできるHTMLを出力してください。",
            "開閉タグや入れ子を整えたHTML5形式で回答を提出してください。",
            "HTML文書として完結するように記述し、構文上の崩れがないことを保証してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "HTML形式で出力してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "HTMLタグ構造を修正し、有効なHTML5文書になるように整えてください。"


def _unwrap_markdown_code_fence(s: str) -> str:
    # Prefer a fenced block explicitly labeled as html
    m = re.search(r"```(?:html)?\s*(.*?)\s*```", s, flags=re.S | re.I)
    if m:
        return m.group(1)
    m = re.search(r"~~~(?:html)?\s*(.*?)\s*~~~", s, flags=re.S | re.I)
    if m:
        return m.group(1)
    return s


__all__ = [
    "HtmlFormatConstraint",
]
